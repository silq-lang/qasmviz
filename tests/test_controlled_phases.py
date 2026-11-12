import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, Parameter
from qiskit.circuit.library import GlobalPhaseGate, XGate
from qiskit.quantum_info import Operator, Statevector, random_statevector
from qiskit_qasm3_import import parse

from qasmviz_basis import parse_extended_basis
from qasmviz_clifford_qasm import dumps_with_phases
from qasmviz_phases import normalize_controlled_phases
from qasmviz_qasm3 import compile_clifford_t_source


ROOT = Path(__file__).resolve().parents[1]
HEADER = 'OPENQASM 3.0; include "stdgates.inc"; '
ADDER = (ROOT / 'tests/circuits/QQ_adder.qasm').read_text()


def postselected_adder(circuit, initial, outcomes):
    """Execute one fixture path without normalizing away its probability/phase.

    Every reset in this fixture immediately follows a measurement. Compare
    unnormalized vectors so branch probabilities and global phases both count.
    """
    outcomes = iter(outcomes)
    classical, measured = {}, {}

    def execute(block, state, wires, bits):
        state = Statevector(state.data * np.exp(1j * float(block.global_phase)))
        for instruction in block.data:
            op = instruction.operation
            qubits = [wires[block.find_bit(q).index] for q in instruction.qubits]
            clbits = [bits[block.find_bit(c).index] for c in instruction.clbits]
            if op.name == 'measure':
                outcome = next(outcomes)
                mask = ((np.arange(len(state.data)) >> qubits[0]) & 1) == outcome
                state = Statevector(state.data * mask)
                classical[clbits[0]] = measured[qubits[0]] = outcome
            elif op.name == 'reset':
                if measured[qubits[0]]:
                    state = state.evolve(XGate(), qubits)
            elif op.name == 'if_else':
                bit, expected = op.condition
                take_true = classical[bits[block.find_bit(bit).index]] == expected
                branch = 0 if take_true else 1
                if branch < len(op.blocks):
                    state = execute(op.blocks[branch], state, qubits, clbits)
            else:
                state = state.evolve(op, qubits)
        return state

    return execute(circuit, initial, list(range(circuit.num_qubits)),
                   list(range(circuit.num_clbits))).data


class ControlledPhaseTests(unittest.TestCase):
    def assertMatrixEqual(self, actual, expected):
        np.testing.assert_allclose(Operator(actual).data, expected, atol=1e-12, rtol=0)

    def test_single_control_is_a_phase_gate_with_exact_global_phase(self):
        for angle, control in itertools.product((0, 0.123, math.pi, math.pi / 4, -math.pi / 4), (0, 1)):
            with self.subTest(angle=angle, control=control):
                modifier = 'ctrl' if control else 'negctrl'
                original = parse(HEADER + f'qubit q; {modifier} @ gphase({angle!r}) q;')
                before = original.copy()
                normalized = normalize_controlled_phases(original)
                diagonal = np.ones(2, dtype=complex)
                diagonal[control] = np.exp(1j * angle)
                self.assertMatrixEqual(normalized, np.diag(diagonal))
                self.assertEqual(normalized.count_ops(), {'p': 1})
                self.assertEqual(original, before)
        normalized = normalize_controlled_phases(parse(HEADER + 'qubit q; negctrl @ gphase(-pi/4) q;'))
        self.assertAlmostEqual(normalized.data[0].operation.params[0], math.pi / 4)

    def test_all_control_patterns_and_wire_orders(self):
        for n in range(2, 5):
            for control in range(1 << n):
                with self.subTest(n=n, control=control):
                    angle = -0.321
                    phase = QuantumCircuit(global_phase=angle).to_gate()
                    original = QuantumCircuit(n)
                    original.append(phase.control(n, ctrl_state=control), reversed(range(n)))
                    normalized = normalize_controlled_phases(original)
                    index = int(f'{control:0{n}b}'[::-1], 2)
                    diagonal = np.ones(1 << n, dtype=complex)
                    diagonal[index] = np.exp(1j * angle)
                    self.assertMatrixEqual(normalized, np.diag(diagonal))
                    self.assertNotIn('circuit-', str(normalized.draw(fold=-1)))

    def test_explicit_global_phase_gate_and_symbolic_parameters(self):
        angle = Parameter('theta')
        original = QuantumCircuit(1, global_phase=0.37)
        original.append(GlobalPhaseGate(angle).control(1, ctrl_state=0), [0])
        normalized = normalize_controlled_phases(original)
        self.assertEqual(normalized.parameters, original.parameters)
        for value in (-0.4, 0.3):
            expected = np.exp(0.37j) * np.diag([np.exp(1j * value), 1])
            self.assertMatrixEqual(normalized.assign_parameters({angle: value}), expected)

    def test_inverse_and_power_inside_control(self):
        for modifiers, angle in (('negctrl @ inv', -math.pi / 4),
                                 ('inv @ negctrl', -math.pi / 4),
                                 ('negctrl @ pow(2)', math.pi / 2)):
            with self.subTest(modifiers=modifiers):
                original = parse(HEADER + f'qubit q; {modifiers} @ gphase(pi/4) q;')
                normalized = normalize_controlled_phases(original)
                self.assertMatrixEqual(normalized, np.diag([np.exp(1j * angle), 1]))
                self.assertEqual(normalized.count_ops(), {'p': 1})

    def test_other_named_and_controlled_gates_are_not_decomposed(self):
        body = QuantumCircuit(1, name='circuit-43', global_phase=0.3)
        body.h(0)
        circuit = QuantumCircuit(2)
        circuit.append(body.to_gate(), [0])
        circuit.append(body.to_gate().control(), [0, 1])
        circuit.cx(0, 1, ctrl_state=0)
        circuit.cp(0.2, 0, 1)
        normalized = normalize_controlled_phases(circuit)
        self.assertEqual(normalized, circuit)
        self.assertMatrixEqual(normalized, Operator(circuit).data)

    def test_nested_control_flow_keeps_phases_in_their_blocks(self):
        source = HEADER + '''bit b; qubit q;
if (b) { negctrl @ gphase(0.2) q; }
else { for int i in [0:1] { if(b) { negctrl @ gphase(-0.3) q; } } }
while(b) { ctrl @ gphase(0.4) q; }
'''
        original = parse(source)
        before = original.copy()
        normalized = normalize_controlled_phases(original)
        self.assertEqual(normalized.global_phase, 0)

        def compare(left, right):
            if any(isinstance(i.operation, ControlFlowOp) for i in left.data):
                for a, b in zip(left.data, right.data, strict=True):
                    for old, new in zip(a.operation.blocks, b.operation.blocks, strict=True):
                        compare(old, new)
            else:
                self.assertMatrixEqual(right, Operator(left).data)
                self.assertNotIn('circuit-', str(right.draw(fold=-1)))

        compare(original, normalized)
        self.assertEqual(original, before)
        self.assertAlmostEqual(normalized.data[0].operation.blocks[0].global_phase, 0.2)

    def test_export_preserves_phase_and_user_gate_definitions(self):
        source = HEADER + '''gate ecr a,b { h a; }
qubit[2] q; ecr q[0],q[1]; negctrl @ gphase(-pi/4) q[0];
'''
        original = parse(source)
        normalized = normalize_controlled_phases(original)
        output = dumps_with_phases(normalized)
        self.assertIn('gphase(', output)
        self.assertMatrixEqual(parse(output), Operator(original).data)

    def test_adder_measurement_branches_preserve_probabilities_and_phases(self):
        original = parse(ADDER)
        normalized = normalize_controlled_phases(original)
        exported = parse(dumps_with_phases(normalized))
        initial = random_statevector(256, seed=941)
        for outcomes in itertools.product((0, 1), repeat=3):
            with self.subTest(outcomes=outcomes):
                expected = postselected_adder(original, initial, outcomes)
                for circuit in (normalized, exported):
                    np.testing.assert_allclose(postselected_adder(circuit, initial, outcomes),
                                               expected, atol=1e-12, rtol=0)

    def test_adder_still_compiles_to_clifford_t(self):
        original = parse(ADDER)
        initial = random_statevector(256, seed=942)
        expected = {outcomes: postselected_adder(original, initial, outcomes)
                    for outcomes in itertools.product((0, 1), repeat=3)}
        for basis in (None, parse_extended_basis('all'), parse_extended_basis('[s,sx,cz]')):
            output = compile_clifford_t_source(ADDER, basis_gates=basis)
            compiled = parse(output)
            for outcomes, state in expected.items():
                with self.subTest(basis=basis, outcomes=outcomes):
                    np.testing.assert_allclose(postselected_adder(compiled, initial, outcomes),
                                               state, atol=1e-11, rtol=0)


class ControlledPhaseCliTests(unittest.TestCase):
    def cli(self, *arguments, source=ADDER):
        return subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *arguments],
                              input=source, text=True, capture_output=True, timeout=30)

    def test_adder_drawing_and_costs_use_recognizable_rotations(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('circuit-', result.stdout)
        self.assertIn('P(π/4)', result.stdout)
        self.assertIn('t-count: 2', result.stdout)
        result = self.cli('--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        costs = json.loads(result.stdout)
        self.assertEqual(costs['t-count'], {'lower': 2, 'upper': 2})
        self.assertEqual(costs['gate-counts']['p'], {'lower': 9, 'upper': 12})

    def test_dump_preserves_branch_phases(self):
        result = self.cli('--dump', '--no-cancel-swaps')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('circuit-', result.stdout)
        self.assertIn('gphase(', result.stdout)
        original, exported = parse(ADDER), parse(result.stdout)
        initial = random_statevector(256, seed=943)
        for outcomes in itertools.product((0, 1), repeat=3):
            np.testing.assert_allclose(postselected_adder(exported, initial, outcomes),
                                       postselected_adder(original, initial, outcomes), atol=1e-12, rtol=0)

    def test_multiple_controls_draw_without_empty_target_error(self):
        source = HEADER + 'qubit[3] q; negctrl @ ctrl(2) @ gphase(pi/4) q[2],q[0],q[1];'
        result = self.cli('--show', '--no-cost', source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('circuit-', result.stdout)
        self.assertIn('P(π/4)', result.stdout)


if __name__ == '__main__':
    unittest.main()
