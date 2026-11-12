from fractions import Fraction
import inspect
import json
from pathlib import Path as FilePath
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import openqasm3
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.classical import expr
from qiskit.circuit.library import XGate, get_standard_gate_name_mapping
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, Statevector

from qasmviz import collect_costs, _dynamic_t_cost
from qasmviz_dynamic_costs import collect_dynamic_costs
from qasmviz_expected import (
    Budget, Event, Limits, NotEasy, ONE, Path, _pull_gate,
    attach_expected_counts, gate, measure, probability_one,
)
from qasmviz_expected_circuit import Executor as CircuitExecutor
from qasmviz_expected_qasm import Executor as SourceExecutor
from qasmviz_qasm3 import compile_clifford_t_source_with_ast


ROOT = FilePath(__file__).resolve().parents[1]
HEADER = 'include "stdgates.inc";\n'


def circuit_costs(circuit):
    options = {name: False for name in inspect.signature(collect_costs).parameters if name != 'circuit'}
    return collect_costs(circuit, **options)


def source_costs(source, **options):
    _, program = compile_clifford_t_source_with_ast(HEADER + source, optimization_level=0, **options)
    return collect_dynamic_costs(program)


def expected(result, metric, name=None):
    value = result['expected-counts'][metric]
    if name is not None:
        value = value[name]
    return None if value is None else Fraction(value['numerator'], value['denominator'])


def pauli(x, z, n):
    return Pauli(([(z >> i) & 1 for i in range(n)], [(x >> i) & 1 for i in range(n)]))


class ExactProbabilityTests(unittest.TestCase):
    def test_sparse_conjugation_matches_every_small_pauli_matrix(self):
        # Independent numerical matrix checks of all exact symbolic rules.
        mapping = get_standard_gate_name_mapping()
        for name in ('h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cy', 'cz',
                     'swap', 'iswap', 'ecr', 'dcx', 't', 'tdg', 'ccx'):
            operation = mapping[name]
            n, matrix = operation.num_qubits, Operator(operation).data
            for x in range(1 << n):
                for z in range(1 << n):
                    with self.subTest(gate=name, x=x, z=z):
                        terms = _pull_gate({(x, z): ONE}, name, tuple(range(n)), Budget())
                        actual = np.zeros_like(matrix)
                        for (a, b), (r, s) in terms.items():
                            actual += (float(r) + np.sqrt(2) * float(s)) * pauli(a, b, n).to_matrix()
                        np.testing.assert_allclose(actual, matrix.conj().T @ pauli(x, z, n).to_matrix() @ matrix,
                                                   atol=2e-14, rtol=0)

    def test_measurement_conditioning_and_reset_match_density_matrices(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        history = tuple(Event('gate', tuple(circuit.find_bit(q).index for q in i.qubits), (i.operation.name,))
                        for i in circuit.data)
        for outcome in (0, 1):
            state = Statevector(circuit).data
            selected = np.array([v if ((i >> 1) & 1) == outcome else 0 for i, v in enumerate(state)])
            selected /= np.linalg.norm(selected)
            rho = DensityMatrix(selected)
            conditional = history + (Event('measure', (1,), (outcome, Fraction(1, 2))),)
            for wire in (0, 1, 2):
                self.assertEqual(probability_one(conditional, wire, Budget()), outcome)
            # Reset one entangled wire without recording its random outcome.
            reset = history + (Event('reset', (0,)),)
            self.assertEqual(probability_one(reset, 0, Budget()), 0)
            self.assertEqual(probability_one(reset, 1, Budget()), Fraction(1, 2))
            after = conditional + (Event('gate', (0,), ('h',)),)
            evolved = rho.evolve(Operator(get_standard_gate_name_mapping()['h']), qargs=[0])
            self.assertAlmostEqual(float(probability_one(after, 0, Budget())), evolved.probabilities([0])[1])

    def test_nonlinear_function_ancilla_has_fair_x_measurement(self):
        history = [Event('gate', (q,), ('unsupported:arbitrary_input',)) for q in range(100)]
        for q in range(99):
            history.append(Event('gate', (q, q + 1, 100), ('ccx',)))
        history.append(Event('gate', (100,), ('h',)))
        budget = Budget()
        self.assertEqual(probability_one(tuple(history), 100, budget), Fraction(1, 2))
        self.assertLess(budget.steps, 250)

    def test_biased_rational_and_irrational_probabilities(self):
        history = (Event('gate', (0,), ('h',)), Event('gate', (1,), ('h',)),
                   Event('gate', (0, 1, 2), ('ccx',)))
        self.assertEqual(probability_one(history, 2, Budget()), Fraction(1, 4))
        history = tuple(Event('gate', (0,), (name,)) for name in ('h', 't', 'h'))
        with self.assertRaisesRegex(NotEasy, 'not rational'):
            probability_one(history, 0, Budget())

    def test_unknown_rotation_is_never_rounded_to_fair(self):
        history = (Event('gate', (0,), ('h',)), Event('gate', (0,), ('rz',)))
        self.assertEqual(probability_one(history, 0, Budget()), Fraction(1, 2))
        with self.assertRaisesRegex(NotEasy, 'unsupported gate'):
            probability_one(history + (Event('gate', (0,), ('h',)),), 0, Budget())

    def test_arbitrary_phases_preserve_the_function_ancilla_proof(self):
        history = (Event('gate', (0, 1), ('unsupported:input_state',)),
                   Event('gate', (0, 1, 2), ('ccx',)),
                   Event('gate', (2,), ('rz',)),
                   Event('gate', (2,), ('h',)))
        self.assertEqual(probability_one(history, 2, Budget()), Fraction(1, 2))


class ExpectedCostTests(unittest.TestCase):
    def test_original_integer_feedforward_and_compiled_gate_counts(self):
        source = (ROOT / 'tests/circuits/classical-feedforward.qasm').read_text()
        for basis in (None, ('x', 'h', 's', 'cx', 't', 'tdg')):
            _, program = compile_clifford_t_source_with_ast(source, optimization_level=0, basis_gates=basis)
            result = collect_dynamic_costs(program)
            self.assertEqual(result['expected-counts']['status'], 'exact')
            self.assertEqual(expected(result, 'measurements'), Fraction(3, 2))
            self.assertEqual(expected(result, 'resets'), Fraction(3, 2))
            self.assertEqual(expected(result, 't-count'), 0)
            if basis is not None:
                self.assertEqual(expected(result, 'gate-counts', 'h'), Fraction(9, 4))
                self.assertEqual(expected(result, 'gate-counts', 'x'), Fraction(1, 4))

    def test_measurement_based_uncomputation_after_actual_t_lowering(self):
        result = source_costs('''qubit[3] q; bit m;
h q[0]; h q[1]; ccx q[0],q[1],q[2];
h q[2]; m=measure q[2]; reset q[2]; if(m) { cz q[0],q[1]; }
''')
        self.assertEqual(result['expected-counts']['status'], 'exact')
        self.assertEqual(expected(result, 't-count'), 7)
        self.assertEqual(expected(result, 'cx-count'), Fraction(13, 2))

    def test_bell_outcomes_and_reused_bits_are_correlated(self):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.h(1)
        circuit.measure([0, 1], [0, 1])
        with circuit.if_test((circuit.clbits[0], True)):
            with circuit.if_test((circuit.clbits[1], True)):
                circuit.t(0)
            with circuit.if_test((circuit.clbits[0], True)):
                circuit.t(1)
        result = circuit_costs(circuit)
        self.assertEqual(expected(result, 't-count'), 1)
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': 2})

    def test_rational_bias_changes_expected_correction_cost(self):
        circuit = QuantumCircuit(3, 1)
        circuit.h(0)
        circuit.h(1)
        circuit.ccx(0, 1, 2)
        circuit.measure(2, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(2)
        result = circuit_costs(circuit)
        self.assertEqual(expected(result, 'gate-counts', 't'), Fraction(1, 4))
        self.assertIsNone(expected(result, 't-count'))  # The existing model leaves CCX's T cost unknown.

    def test_negative_controls_preserve_function_ancilla_probabilities(self):
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.append(XGate().control(1, ctrl_state=0), [0, 1])
        circuit.h(1)
        circuit.measure(1, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        self.assertEqual(expected(circuit_costs(circuit), 't-count'), Fraction(1, 2))

    def test_classical_assignments_arrays_scopes_and_fixed_loop(self):
        result = source_costs('''qubit q; bit[2] m; int[64] n=0;
for int i in [0:1] {
  h q; m[i]=measure q; reset q;
  if(m[i]) { n += 1; }
}
if(n == 2) { int n=100; t q; }
if(n == 0) { t q; }
''')
        self.assertEqual(result['expected-counts']['status'], 'exact')
        self.assertEqual(expected(result, 't-count'), Fraction(1, 2))

    def test_conditional_break_continue_and_switch(self):
        result = source_costs('''qubit q; bit b; int n=0;
for int i in [0:2] {
  h q; b=measure q; reset q;
  if(b) { break; }
  n += 1; continue; t q;
}
switch(n) { case 0 { t q; } case 1 { t q; t q; } default { h q; } }
''')
        self.assertEqual(expected(result, 'measurements'), Fraction(7, 4))
        self.assertEqual(expected(result, 't-count'), 1)

    def test_circuit_classical_store_and_nontrivial_block_wire_mapping(self):
        circuit = QuantumCircuit(4, 3)
        circuit.h(3)
        circuit.measure(3, 2)
        circuit.store(circuit.clbits[0], expr.logic_not(circuit.clbits[2]))
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(1)
            circuit.measure(3, 1)
            with circuit.if_test((circuit.clbits[1], True)):
                circuit.t(0)
        self.assertEqual(expected(circuit_costs(circuit), 't-count'), Fraction(1, 2))

    def test_separately_constructed_blocks_remap_wires_and_bits(self):
        circuit = QuantumCircuit(4, 3)
        circuit.h(3)
        circuit.measure(3, 2)
        body = QuantumCircuit(2, 2)
        body.measure(0, 1)
        with body.if_test((body.clbits[1], True)):
            body.t(1)
        circuit.if_else((circuit.clbits[2], True), body, None, [3, 1], [2, 0])
        self.assertEqual(expected(circuit_costs(circuit), 't-count'), Fraction(1, 2))

    def test_adaptive_quantum_corrections_match_exhaustive_statevectors(self):
        # Enumerate physical outcome probabilities independently of the sparse
        # Pauli implementation, including feedback before the next measurement.
        rng = np.random.default_rng(719)
        mapping = get_standard_gate_name_mapping()
        for _ in range(12):
            circuit = QuantumCircuit(2, 1)
            states = [(1.0, Statevector.from_label('00'), 0)]
            for stage in range(3):
                for _ in range(3):
                    name = str(rng.choice(['h', 's', 'sx', 'cx']))
                    wires = [0, 1] if name == 'cx' else [int(rng.integers(2))]
                    operation = mapping[name]
                    circuit.append(operation, wires)
                    states = [(p, state.evolve(operation, qargs=wires), cost) for p, state, cost in states]
                wire = int(rng.integers(2))
                circuit.measure(wire, 0)
                with circuit.if_test((circuit.clbits[0], True)):
                    circuit.x(1 - wire)
                    circuit.s(wire)
                children = []
                for probability, state, cost in states:
                    for outcome in (0, 1):
                        vector = np.array([amplitude if ((i >> wire) & 1) == outcome else 0
                                           for i, amplitude in enumerate(state.data)])
                        p = float(np.vdot(vector, vector).real)
                        if p < 1e-12:
                            continue
                        projected = Statevector(vector / np.sqrt(p))
                        if outcome:
                            projected = projected.evolve(mapping['x'], [1 - wire]).evolve(mapping['s'], [wire])
                        children.append((probability * p, projected, cost + 2 * outcome))
                states = children
            result = circuit_costs(circuit)
            self.assertAlmostEqual(float(expected(result, 'gate-count')),
                                   9 + sum(p * cost for p, _, cost in states))

    def test_qiskit_counted_loop_retains_measurement_correlations(self):
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop(range(4)):
            with circuit.if_test((circuit.clbits[0], True)):
                circuit.t(0)
        self.assertEqual(expected(circuit_costs(circuit), 't-count'), 2)

    def test_unknown_input_and_irrational_probability_keep_bounds(self):
        for prefix, reason in (('', 'unknown'), ('h q; t q; h q; b=measure q;', 'not rational')):
            result = source_costs('qubit q; bit b; ' + prefix + ' if(b) { t q; }')
            self.assertIsNone(expected(result, 't-count'))
            self.assertIn(reason, result['expected-counts']['reason'])
            self.assertIsNotNone(expected(result, 'measurements'))
            self.assertIsNone(result['depth'])
            self.assertIsNone(result['t-depth'])

    def test_same_named_custom_gate_cannot_masquerade_as_h(self):
        body = QuantumCircuit(1)
        body.x(0)
        fake_h = body.to_gate()
        fake_h.name = 'h'
        circuit = QuantumCircuit(1, 1)
        circuit.append(fake_h, [0])
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        result = circuit_costs(circuit)
        self.assertIsNone(expected(result, 't-count'))
        self.assertIn('unsupported:h', result['expected-counts']['reason'])

    def test_unknown_instructions_are_not_assumed_to_be_unitary(self):
        circuit = QuantumCircuit(2, 1)
        circuit.append(Instruction('opaque', 1, 0, []), [1])
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        result = circuit_costs(circuit)
        self.assertIsNone(expected(result, 'gate-counts', 't'))
        self.assertIn('unsupported quantum instruction', result['expected-counts']['reason'])


class ExpectedCostBudgetTests(unittest.TestCase):
    def test_fixed_bounds_skip_execution_even_for_huge_loops(self):
        result = {'width': 1000000, 't-count': {'lower': 10**12, 'upper': 10**12}, 'gate-counts': {}}
        def unexpected(_):
            self.fail('fixed costs must not invoke the executor')
        attach_expected_counts(result, unexpected)
        self.assertEqual(expected(result, 't-count'), 10**12)

    def test_unsupported_loops_are_rejected_before_probability_queries(self):
        program = openqasm3.parse(HEADER + 'qubit q; bit b; h q; b=measure q; while(b) { t q; }')
        with patch('qasmviz_expected.probability_one', side_effect=AssertionError('probability queried')):
            with self.assertRaisesRegex(NotEasy, 'WhileLoop'):
                SourceExecutor(program, Budget()).run()
            circuit = QuantumCircuit(1, 1)
            circuit.h(0)
            circuit.measure(0, 0)
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.t(0)
            with self.assertRaisesRegex(NotEasy, 'while loop'):
                CircuitExecutor(circuit, _dynamic_t_cost, Budget()).run()
            program = openqasm3.parse(HEADER + 'qubit q; bit b; h q; b=measure q; for int i in [0:1000000000000] { if(b) { t q; } }')
            with self.assertRaisesRegex(NotEasy, 'iteration limit'):
                SourceExecutor(program, Budget()).run()

    def test_path_limit_discards_partial_totals(self):
        circuit = QuantumCircuit(1, 1)
        for _ in range(8):
            circuit.h(0)
            circuit.measure(0, 0)
            circuit.reset(0)
            with circuit.if_test((circuit.clbits[0], True)):
                circuit.t(0)
        result = circuit_costs(circuit)
        self.assertIsNone(expected(result, 't-count'))
        self.assertEqual(expected(result, 'measurements'), 8)
        self.assertIn('limit reached', result['expected-counts']['reason'])

    def test_each_quantum_limit_is_checked_before_growth(self):
        history = tuple(Event('gate', (0,), (name,)) for name in ('h', 't', 'h'))
        budget = Budget(Limits(steps=1))
        with self.assertRaisesRegex(NotEasy, 'work limit'):
            probability_one(history, 0, budget)
        self.assertLessEqual(budget.steps, 1)
        with self.assertRaisesRegex(NotEasy, 'expression limit'):
            probability_one(history, 0, Budget(Limits(terms=1)))
        path = Path()
        with self.assertRaisesRegex(NotEasy, 'history limit'):
            gate(path, 'h', (0,), Budget(Limits(history=0)), 0)
        self.assertEqual(path.history, ())

    def test_history_and_loop_limits_keep_existing_bounds(self):
        circuit = QuantumCircuit(1, 1)
        for _ in range(513):
            circuit.h(0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.t(0)
        result = circuit_costs(circuit)
        self.assertIn('history limit', result['expected-counts']['reason'])
        self.assertIsNone(expected(result, 't-count'))
        circuit = QuantumCircuit(1, 1)
        with circuit.for_loop(range(10**12)):
            with circuit.if_test((circuit.clbits[0], True)):
                circuit.t(0)
        result = circuit_costs(circuit)
        self.assertIn('iteration limit', result['expected-counts']['reason'])

    def test_cli_text_and_json_have_expected_counts_but_no_expected_depth(self):
        source = HEADER + 'qubit q; bit b; h q; b=measure q; if(b) { t q; }'
        for flags in (['--json'], ['--clifford-t', '--json'], ['--clifford-t-ext=[s,sx,cz]', '--json'], ['--cost']):
            process = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *flags], input=source,
                                     text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            if '--json' in flags:
                result = json.loads(process.stdout)
                self.assertEqual(expected(result, 't-count'), Fraction(1, 2))
                self.assertNotIn('depth', result['expected-counts'])
            else:
                self.assertIn('t-count: 1/2', process.stdout)
                self.assertIn('Depth is not estimated', process.stdout)


if __name__ == '__main__':
    unittest.main()
