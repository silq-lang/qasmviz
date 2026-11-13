from fractions import Fraction
import json
import math
from pathlib import Path as FilePath
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import openqasm3
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.classical import expr
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import DensityMatrix, Operator, Pauli

from qasmviz import _dynamic_t_cost
from qasmviz_expected import Budget, Event, Limits, NotEasy, ONE, Path, _pull_event, canonical_phase
from qasmviz_expected_circuit import Executor as CircuitExecutor
from qasmviz_expected_qasm import Executor as SourceExecutor
from qasmviz_qasm3 import compile_clifford_t_source_with_ast


ROOT = FilePath(__file__).resolve().parents[1]


def adder(n):
    """The reported ripple adder, with n input bits per register."""
    circuit = QuantumCircuit(3 * n + 1, n - 1)
    circuit.h(range(2 * n + 1))
    for j in range(n - 1):
        a, b, t = j, n + j, 2 * n + j
        if j:
            circuit.h(t)
        circuit.p(math.pi / 4, t)
        circuit.cx(b, t)
        circuit.cx(a, t)
        pairs = ((b, t + 1), (t, t + 1), (a, t + 2), (t, t + 2))
        for control, target in pairs:
            circuit.cx(control, target)
        circuit.p(-math.pi / 4, t + 1)
        circuit.p(-math.pi / 4, t + 2)
        circuit.p(math.pi / 4, t)
        for control, target in pairs:
            circuit.cx(control, target)
        circuit.h(t)
        circuit.p(math.pi / 2, t)
        if j:
            circuit.cx(t - 1, t)
        if j < n - 2:
            circuit.cx(t, b + 1)
            circuit.cx(t, a + 1)
        else:
            circuit.cx(t, n - 1)
            circuit.cx(2 * n - 1, n - 1)
    for j in reversed(range(n - 1)):
        a, b, t = j, n + j, 2 * n + j
        if j:
            circuit.cx(t - 1, t)
        circuit.h(t)
        circuit.measure(t, n - 2 - j)
        circuit.reset(t)
        with circuit.if_test((circuit.clbits[n - 2 - j], True)):
            circuit.cp(math.pi, b, a)
        if j:
            circuit.cx(t - 1, b)
        circuit.cx(b, a)
    return circuit


class ExpectedScalingTests(unittest.TestCase):
    def test_reported_adder_through_all_cli_paths(self):
        source = (ROOT / 'tests/circuits/adder-31.qasm').read_text()
        for flags in ([], ['--clifford-t'], ['--clifford-t-ext']):
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), '--json', *flags],
                                        input=source, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = json.loads(result.stdout)['expected-counts']
                self.assertEqual(expected['status'], 'exact')
                self.assertEqual(expected['t-count'], {'numerator': 36, 'denominator': 1})
                self.assertEqual(expected['measurements'], {'numerator': 9, 'denominator': 1})
                self.assertEqual(expected['resets'], {'numerator': 9, 'denominator': 1})
                if not flags:
                    self.assertEqual(expected['gate-counts']['cp'], {'numerator': 9, 'denominator': 2})
                    self.assertEqual(expected['2q-count'], {'numerator': 291, 'denominator': 2})
                self.assertNotIn('depth', expected)

    def test_adder_work_scales_linearly_without_branching_or_global_queries(self):
        previous = None
        for n in (10, 20, 40, 80):
            budget = Budget()
            # Every measurement must be proved by local forward facts. This
            # would reject an implementation that repeatedly scans the prefix.
            with patch('qasmviz_expected_stream.probability_one', side_effect=AssertionError('global query')):
                paths = CircuitExecutor(adder(n), _dynamic_t_cost, budget).run()
            self.assertEqual(len(paths), 1)
            self.assertEqual(budget.paths, 1)
            self.assertEqual(paths[0].counts['t-count'], 4 * (n - 1))
            self.assertEqual(paths[0].counts['gate:cp'], Fraction(n - 1, 2))
            self.assertLess(budget.steps, 650 * n)
            self.assertEqual(budget.step_limit, 256 + 64 * budget.input_size)
            if previous is not None:
                self.assertLess(budget.steps, 2.2 * previous)
            previous = budget.steps

    def test_compiler_scheduling_preserves_local_proofs(self):
        source = (ROOT / 'tests/circuits/adder-31.qasm').read_text()
        _, program = compile_clifford_t_source_with_ast(source, optimization_level=0)
        budget = Budget()
        with patch('qasmviz_expected_stream.probability_one', side_effect=AssertionError('global query')):
            paths = SourceExecutor(program, budget).run()
        self.assertEqual(paths[0].counts['t-count'], 36)
        self.assertEqual(budget.paths, 1)
        self.assertLess(budget.steps, 15000)

    def test_many_independent_results_reuse_one_bit(self):
        circuit = QuantumCircuit(1, 1)
        for _ in range(200):
            circuit.h(0)
            circuit.measure(0, 0)
            circuit.reset(0)
            with circuit.if_test((circuit.clbits[0], True)):
                circuit.t(0)
        for source in (False, True):
            budget = Budget()
            if source:
                _, program = compile_clifford_t_source_with_ast(qasm3.dumps(circuit), optimization_level=0)
                paths = SourceExecutor(program, budget).run()
            else:
                paths = CircuitExecutor(circuit, _dynamic_t_cost, budget).run()
            self.assertEqual(paths[0].counts['gate:t'], 100)
            self.assertEqual(budget.paths, 1)

    def test_declared_register_sizes_contribute_to_the_budget(self):
        program = openqasm3.parse('qubit[2048] q; bit[2048] m; h q[0]; m[0]=measure q[0]; if(m[0]) { t q[0]; }')
        budget = Budget()
        paths = SourceExecutor(program, budget).run()
        self.assertEqual(paths[0].counts['gate:t'], Fraction(1, 2))
        self.assertGreaterEqual(budget.input_size, 4096)

    def test_feedback_changes_later_marginals(self):
        # Bell-correlated results cannot be treated as independent coins.
        # The first correction makes the second measurement certainly zero.
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.reset(0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.x(1)
        circuit.measure(1, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        for source in (False, True):
            budget = Budget()
            if source:
                _, program = compile_clifford_t_source_with_ast(qasm3.dumps(circuit), optimization_level=0)
                paths = SourceExecutor(program, budget).run()
            else:
                paths = CircuitExecutor(circuit, _dynamic_t_cost, budget).run()
            self.assertEqual(paths[0].counts.get('gate:t', 0), 0)
            self.assertEqual(budget.paths, 1)

    def test_unused_measurement_still_dephases_state(self):
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)  # overwritten without a read
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        paths = CircuitExecutor(circuit, _dynamic_t_cost, Budget()).run()
        self.assertEqual(paths[0].counts['gate:t'], Fraction(1, 2))

    def test_negated_guard_and_else_costs(self):
        source = '''include "stdgates.inc";
qubit[3] q; bit b;
h q[0]; h q[1]; ccx q[0],q[1],q[2];
b=measure q[2]; reset q[2];
if(!(b==true)) { t q[0]; } else { t q[0]; t q[1]; t q[2]; }
'''
        # The AND is true with probability 1/4: 3/4 * 1 + 1/4 * 3.
        circuit = QuantumCircuit(3, 1)
        circuit.h([0, 1])
        circuit.ccx(0, 1, 2)
        circuit.measure(2, 0)
        circuit.reset(2)
        with circuit.if_test(expr.logic_not(expr.equal(circuit.clbits[0], True))) as otherwise:
            circuit.t(0)
        with otherwise:
            circuit.t([0, 1, 2])
        paths = CircuitExecutor(circuit, _dynamic_t_cost, Budget()).run()
        self.assertEqual(paths[0].counts['gate:t'], Fraction(3, 2))
        _, program = compile_clifford_t_source_with_ast(source, optimization_level=0)
        paths = SourceExecutor(program, Budget()).run()
        self.assertEqual(paths[0].counts['t-count'], 7 + Fraction(3, 2))

    def test_feedback_channel_matches_kraus_matrices(self):
        def matrix(x, z):
            return Pauli(([(z >> i) & 1 for i in range(2)],
                          [(x >> i) & 1 for i in range(2)])).to_matrix()
        branches = ((Event('gate', (1,), ('h',)), Event('gate', (1,), ('t',))),
                    (Event('gate', (0, 1), ('cx',)), Event('gate', (0,), ('s',))))
        unitaries = []
        mapping = get_standard_gate_name_mapping()
        for branch in branches:
            circuit = QuantumCircuit(2)
            for event in branch:
                circuit.append(mapping[event.value[0]], event.wires)
            unitaries.append(Operator(circuit).data)
        for reset in (False, True):
            common = (Event('reset', (0,)),) if reset else ()
            event = Event('instrument', (0, 1), (common, branches))
            kraus = []
            for outcome, unitary in enumerate(unitaries):
                local = np.zeros((2, 2))
                local[0 if reset else outcome, outcome] = 1
                kraus.append(unitary @ np.kron(np.eye(2), local))
            for x in range(4):
                for z in range(4):
                    terms = _pull_event({(x, z): ONE}, event, Budget())
                    actual = sum((float(a) + math.sqrt(2) * float(b)) * matrix(s, t)
                                 for (s, t), (a, b) in terms.items())
                    reference = sum(k.conj().T @ matrix(x, z) @ k for k in kraus)
                    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=0)

    def test_only_exact_canonical_float_literals_are_recognized(self):
        for k in range(-8, 9):
            angle = k * math.pi / 4
            self.assertIsNotNone(canonical_phase('p', (0,), angle))
            self.assertIsNone(canonical_phase('p', (0,), math.nextafter(angle, math.inf)))
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.p(math.nextafter(math.pi / 2, math.inf), 0)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        with self.assertRaisesRegex(NotEasy, 'unsupported gate'):
            CircuitExecutor(circuit, _dynamic_t_cost, Budget()).run()

    def test_forward_facts_agree_with_density_matrices(self):
        rng, mapping = np.random.default_rng(9417), get_standard_gate_name_mapping()
        path, budget, state = Path(), Budget(Limits(steps=100000)), DensityMatrix.from_label('0000')
        for _ in range(400):
            name = str(rng.choice(('h', 'x', 's', 't', 'tdg', 'cx', 'ccx', 'reset')))
            arity = 3 if name == 'ccx' else 2 if name == 'cx' else 1
            wires = tuple(int(q) for q in rng.choice(4, arity, replace=False))
            event = Event('reset', wires) if name == 'reset' else Event('gate', wires, (name,))
            path.record(event, budget)
            state = state.reset(wires) if name == 'reset' else state.evolve(mapping[name], qargs=wires)
            for q in range(4):
                probability = path.facts.probability(q)
                if probability is not None:
                    self.assertAlmostEqual(float(probability), state.probabilities([q])[1], places=12)

    def test_changing_a_function_witness_invalidates_the_relation(self):
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)  # q[1] is now |+>, no longer a function of q[0].
        circuit.h(1)
        circuit.measure(1, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        paths = CircuitExecutor(circuit, _dynamic_t_cost, Budget()).run()
        self.assertEqual(paths[0].counts.get('gate:t', 0), 0)

    def test_explicit_work_cap_and_expression_cap_still_stop_early(self):
        budget = Budget(Limits(steps=100))
        with self.assertRaisesRegex(NotEasy, 'work limit'):
            CircuitExecutor(adder(10), _dynamic_t_cost, budget).run()
        self.assertLessEqual(budget.steps, 100)
        # A genuinely hard observable must not trigger an unbounded expansion.
        circuit = QuantumCircuit(8, 1)
        for q in range(8):
            circuit.h(q)
        for q in range(1, 8):
            circuit.cx(q, 0)
        for q in range(8):
            circuit.t(q)
            circuit.h(q)
        for q in range(1, 8):
            circuit.cx(q, 0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        budget = Budget()
        with self.assertRaisesRegex(NotEasy, 'expression limit'):
            CircuitExecutor(circuit, _dynamic_t_cost, budget).run()
        self.assertLess(budget.steps, 2000)


if __name__ == '__main__':
    unittest.main()
