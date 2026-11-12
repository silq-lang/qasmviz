import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_qasm3_import import parse

from qasmviz_layout import restore_layout


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'include "stdgates.inc"; qubit[4] q; h q[0]; swap q[0],q[1];'


class LayoutTests(unittest.TestCase):
    def test_initial_and_final_permutations_preserve_every_input_state(self):
        circuit = QuantumCircuit(4, global_phase=0.37)
        circuit.h(0)
        circuit.cx(0, 3)
        circuit.swap(1, 3)
        circuit.s(2)
        for basis in (['cx', 'u'], ['cz', 'sx', 'rz'], ['rxx', 'rx', 'rz']):
            compiled = generate_preset_pass_manager(
                3, basis_gates=basis, initial_layout=[2, 0, 3, 1],
                qubits_initially_zero=False, seed_transpiler=777,
            ).run(circuit)
            restored = restore_layout(compiled, basis_gates=basis)
            self.assertIsNone(restored.layout)
            self.assertLessEqual(set(restored.count_ops()), set(basis))
            np.testing.assert_allclose(Operator(restored).data, Operator(circuit).data, atol=1e-12, rtol=0)
            self.assertIsNotNone(compiled.layout)

    def test_physical_restoration_respects_directed_couplings(self):
        backend = GenericBackendV2(4, basis_gates=['cx', 'rz', 'sx', 'x'],
                                   coupling_map=[[0, 1], [1, 2], [2, 3]], seed=7)
        original = QuantumCircuit(4, global_phase=0.23)
        original.h(0)
        original.cx(0, 3)
        original.swap(1, 3)
        original.s(2)
        compiled = generate_preset_pass_manager(
            3, target=backend.target, initial_layout=[2, 0, 3, 1],
            qubits_initially_zero=False, seed_transpiler=3,
        ).run(original)
        for zero in (False, True):
            restored = restore_layout(compiled, target=backend.target, initially_zero=zero)
            for instruction in restored.data:
                wires = tuple(restored.find_bit(q).index for q in instruction.qubits)
                self.assertTrue(backend.target.instruction_supported(instruction.operation.name, wires))
            if zero:
                np.testing.assert_allclose(Statevector(restored).data, Statevector(original).data, atol=1e-12)
            else:
                np.testing.assert_allclose(Operator(restored).data, Operator(original).data, atol=1e-12)

    def test_backend_ancillas_and_idle_wires_are_retained(self):
        backend = GenericBackendV2(4, basis_gates=['cx', 'rz', 'sx', 'x'],
                                   coupling_map=[[0, 1], [1, 2], [2, 3]], seed=8)
        original = QuantumCircuit(2)
        original.x(0)
        original.swap(0, 1)
        compiled = generate_preset_pass_manager(
            3, target=backend.target, initial_layout=[2, 3], seed_transpiler=7,
        ).run(original)
        restored = restore_layout(compiled, target=backend.target, initially_zero=True)
        expected = QuantumCircuit(4)
        expected.x(1)
        self.assertEqual(restored.num_qubits, 4)
        self.assertIsNone(restored.layout)
        self.assertTrue(Statevector(restored).equiv(Statevector(expected)))

    def test_cli_dump_run_and_cost_include_restoration(self):
        targets = ('--clifford-t', '--clifford-t-ext', '--cx-u', '--cz-sx', '--ibm-heron')
        expected = Statevector(parse(SOURCE))
        for target, output in itertools.product(targets, ('--dump', '--run', '--json')):
            with self.subTest(target=target, output=output):
                arguments = [sys.executable, str(ROOT / 'qasmviz.py'), target, '--no-cancel-swaps', output]
                if output == '--run':
                    arguments.append('--json')
                result = subprocess.run(arguments, input=SOURCE, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                if output == '--dump':
                    imported = parse(result.stdout)
                    self.assertEqual(imported.num_qubits, 4)
                    self.assertTrue(expected.equiv(Statevector(imported)))
                elif output == '--run':
                    state = np.zeros(16, dtype=complex)
                    for value in json.loads(result.stdout)['statevector']:
                        state[int(value['basis'][::-1], 2)] = complex(value['re'], value['im'])
                    self.assertTrue(expected.equiv(Statevector(state)))
                else:
                    costs = json.loads(result.stdout)
                    self.assertEqual(costs['width'], 4)
                    self.assertGreater(costs['gates']['total_count'], 1)

    def test_classical_readout_order_survives_restoration(self):
        source = 'include "stdgates.inc"; qubit[2] q; bit[2] m; x q[0]; swap q[0],q[1]; m=measure q;'
        for target in ('--clifford-t', '--cx-u'):
            result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), target,
                                     '--no-cancel-swaps', '--run', '--json'],
                                    input=source, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['clbits'], '01')


if __name__ == '__main__':
    unittest.main()
