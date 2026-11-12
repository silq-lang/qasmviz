import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, qasm2
from qiskit.circuit import Parameter
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator, Statevector
from qiskit_qasm3_import import parse

from qasmviz_qasm2 import dumps_qasm2


ROOT = Path(__file__).resolve().parents[1]


class Qasm2ExportTests(unittest.TestCase):
    def roundtrip(self, circuit):
        before = circuit.copy()
        source = dumps_qasm2(circuit)
        actual = qasm2.loads(source, strict=True)
        self.assertEqual(actual.num_qubits, circuit.num_qubits)
        np.testing.assert_allclose(Operator(actual).data, Operator(circuit).data, atol=2e-12, rtol=0)
        self.assertEqual(circuit, before)
        return source

    def test_p_and_cp_use_exact_qelib1_equivalents(self):
        circuit = QuantumCircuit(2)
        circuit.p(0.37, 0)
        circuit.cp(0.71, 0, 1)
        circuit.u(0.2, 0.3, 0.4, 1)
        source = self.roundtrip(circuit)
        self.assertIn('u1(0.37)', source)
        self.assertIn('cu1(0.71)', source)
        self.assertIn('u3(0.2,0.3,0.4)', source)
        self.assertNotIn('rz(', source)

    def test_standard_gates_missing_from_strict_qelib1_have_definitions(self):
        mapping = get_standard_gate_name_mapping()
        for name in ('sx', 'sxdg', 'swap', 'cswap', 'dcx', 'iswap', 'ecr', 'rxx', 'ryy',
                     'rzz', 'rzx', 'cs', 'csdg', 'ccz', 'cu', 'crx', 'cry'):
            with self.subTest(gate=name):
                gate = mapping[name]
                circuit = QuantumCircuit(gate.num_qubits, global_phase=0.17)
                circuit.append(gate, circuit.qubits)
                circuit = circuit.assign_parameters({p: 0.321 for p in circuit.parameters})
                source = self.roundtrip(circuit)
                self.assertIn('gate ' + name, source)

    def test_definition_phases_and_controlled_uses_are_exact(self):
        body = QuantumCircuit(1, name='phaseful', global_phase=0.37)
        body.p(0.12, 0)
        body.h(0)
        circuit = QuantumCircuit(3, global_phase=0.43)
        circuit.append(body.to_gate(), [0])
        circuit.append(body.to_gate().control(1), [1, 0])
        circuit.append(body.to_gate().control(2, ctrl_state=1), [2, 1, 0])
        self.roundtrip(circuit)

    def test_same_named_definitions_and_register_names_do_not_collide(self):
        circuit = QuantumCircuit(QuantumRegister(2, 'qasmviz_export_gate'))
        for angle in (0.2, 0.3, 0.2):
            body = QuantumCircuit(1, name='sx', global_phase=angle)
            body.rx(angle, 0)
            circuit.append(body.to_gate(), [0])
        circuit.sx(1)
        self.roundtrip(circuit)

    def test_idle_wires_and_custom_bound_parameters_survive(self):
        a = Parameter('theta')
        body = QuantumCircuit(1, name='my_gate')
        body.rz(a, 0)
        circuit = QuantumCircuit(5, global_phase=0.7)
        circuit.append(body.assign_parameters({a: 0.2}).to_gate(), [2])
        self.roundtrip(circuit)

    def test_unbound_parameters_and_phase_without_qubits_fail_explicitly(self):
        circuit = QuantumCircuit(1)
        circuit.p(Parameter('a'), 0)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, 'bound'):
            dumps_qasm2(circuit)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, 'without a qubit'):
            dumps_qasm2(QuantumCircuit(global_phase=0.2))

    def test_unitary_conditional_body_is_one_qasm2_gate(self):
        source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; if(c==1) sx q[0];'
        # sx is a Qiskit extension on input; the output must define it itself.
        original = qasm2.loads(source, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        output = dumps_qasm2(original)
        exported = qasm2.loads(output, strict=True)
        before, after = original.data[0].operation.blocks[0], exported.data[0].operation.blocks[0]
        np.testing.assert_allclose(Operator(before).data, Operator(after).data, atol=1e-12)

    def test_cli_export_is_strict_and_phase_preserving(self):
        source = 'include "stdgates.inc"; qubit[3] q; gphase(0.31); p(0.37) q[0]; ctrl @ p(0.71) q[0],q[1]; sx q[1];'
        result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), '--dump2', '--no-cancel-swaps'],
                                input=source, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = qasm2.loads(result.stdout, strict=True)
        np.testing.assert_allclose(Operator(actual).data, Operator(parse(source)).data, atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec('cirq'), 'Cirq is optional')
    def test_cirq_accepts_export_with_exact_phase(self):
        import cirq
        from cirq.contrib.qasm_import import circuit_from_qasm
        circuit = QuantumCircuit(2, global_phase=0.23)
        circuit.h(0)
        circuit.p(0.3, 1)
        circuit.cp(0.71, 0, 1)
        circuit.ecr(0, 1)
        converted = circuit_from_qasm(dumps_qasm2(circuit))
        np.testing.assert_allclose(cirq.unitary(converted), Operator(circuit).reverse_qargs().data,
                                   atol=1e-12, rtol=0)

    @unittest.skipUnless(importlib.util.find_spec('pytket'), 'pytket is optional')
    def test_pytket_accepts_export_with_exact_phase(self):
        from pytket.qasm import circuit_from_qasm_str
        circuit = QuantumCircuit(2, global_phase=0.23)
        circuit.h(0)
        circuit.p(0.3, 1)
        circuit.cp(0.71, 0, 1)
        circuit.ecr(0, 1)
        converted = circuit_from_qasm_str(dumps_qasm2(circuit))
        np.testing.assert_allclose(converted.get_unitary(), Operator(circuit).reverse_qargs().data,
                                   atol=1e-12, rtol=0)

    @unittest.skipUnless(importlib.util.find_spec('cirq'), 'Cirq is optional')
    def test_cirq_adapter_preserves_idle_wires_and_numeric_order(self):
        from qasmviz import _compile_cirq
        circuit = QuantumCircuit(12)
        circuit.x(2)
        circuit.h(10)
        _, selected, _ = _compile_cirq(circuit, gateset_name='cz')
        self.assertEqual(selected.num_qubits, 12)
        self.assertTrue(Statevector(circuit).equiv(Statevector(selected)))

    @unittest.skipUnless(importlib.util.find_spec('cirq'), 'Cirq is optional')
    def test_cirq_adapter_preserves_unused_classical_bits_and_readout_order(self):
        from qasmviz import _compile_cirq, format_clbits, run_circuit_and_capture
        circuit = QuantumCircuit(3, 3)
        circuit.x(2)
        circuit.measure(2, 2)
        _, selected, _ = _compile_cirq(circuit, gateset_name='cz')
        self.assertEqual((selected.num_qubits, selected.num_clbits), (3, 3))
        classical, _ = run_circuit_and_capture(selected)
        self.assertEqual(format_clbits(classical), '001')

    @unittest.skipUnless(importlib.util.find_spec('pytket'), 'pytket is optional')
    def test_pytket_adapters_preserve_phases_and_implicit_permutations(self):
        from qasmviz import _compile_pytket, _compile_pytket_tk2
        from pytket import OpType
        from pytket.passes import AutoRebase
        circuit = QuantumCircuit(3, global_phase=0.37)
        circuit.h(0)
        circuit.cp(0.3, 0, 1)
        circuit.p(0.4, 1)
        circuit.swap(0, 2)
        for target in ('tk2', 'iqm'):
            if target == 'tk2':
                _, selected, _ = _compile_pytket_tk2(circuit)
            else:
                _, selected, _ = _compile_pytket(circuit, rebase_pass=AutoRebase({OpType.PhasedX, OpType.CZ, OpType.Rz}),
                                                 gateset_name='iqm', phasedx_gate_name='prx')
            np.testing.assert_allclose(Operator(selected).data, Operator(circuit).data, atol=1e-12)
            self.roundtrip(selected)

    @unittest.skipUnless(importlib.util.find_spec('cirq') and importlib.util.find_spec('pytket'), 'optional backends')
    def test_native_target_cli_exports_both_qasm_versions(self):
        source = 'include "stdgates.inc"; qubit[3] q; h q[0]; ctrl @ p(0.71) q[0],q[1]; swap q[0],q[2];'
        for target in ('--quera-gemini', '--tk2', '--iqm'):
            for output in ('--dump', '--dump2'):
                with self.subTest(target=target, output=output):
                    result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), target, output, '--no-cancel-swaps'],
                                            input=source, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    imported = parse(result.stdout) if output == '--dump' else qasm2.loads(result.stdout, strict=True)
                    self.assertEqual(imported.num_qubits, 3)
                    self.assertTrue(Operator(imported).equiv(Operator(parse(source))))


if __name__ == '__main__':
    unittest.main()
