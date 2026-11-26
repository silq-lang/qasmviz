import inspect
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qiskit import QuantumCircuit
from qiskit.circuit import CASE_DEFAULT, Parameter
from qiskit.circuit.classical import expr
from qiskit.circuit.library import CPhaseGate, CU1Gate, CZGate
from qiskit.quantum_info import Operator
from qiskit_qasm3_import import parse

from qasmviz import _dynamic_t_cost, collect_costs
from qasmviz_circuit_costs import collect_circuit_dynamic_costs


ROOT = Path(__file__).resolve().parents[1]
HEADER = 'include "stdgates.inc"; '


def costs(circuit):
    options = {name: False for name in inspect.signature(collect_costs).parameters if name != 'circuit'}
    return collect_costs(circuit, **options)


class CircuitDynamicCostTests(unittest.TestCase):
    def assertBound(self, result, key, lower, upper):
        self.assertEqual(result[key], {'lower': lower, 'upper': upper})

    def test_conditionals_are_not_two_qubit_gates_and_do_not_hide_t(self):
        circuit = parse(HEADER + '''qubit[2] q; bit b;
cx q[0],q[1]; b=measure q[0]; if(b) { h q[0]; h q[1]; t q[1]; }
''')
        result = costs(circuit)
        self.assertBound(result, '2q-count', 1, 1)
        self.assertBound(result, 'gate-count', 1, 4)
        self.assertBound(result, 't-count', 0, 1)
        self.assertBound(result, 'measurements', 1, 1)
        self.assertNotIn('if_else', result['gate-counts'])
        self.assertIsNone(result['depth'])
        self.assertIsNone(result['t-depth'])

    def test_alternative_branches_are_not_summed(self):
        circuit = parse(HEADER + '''qubit[2] q; bit b;
if(b) { t q[0]; } else { t q[1]; t q[1]; t q[1]; reset q[0]; }
''')
        result = costs(circuit)
        self.assertBound(result, 'gate-count', 1, 3)
        self.assertBound(result, 't-count', 1, 3)
        self.assertBound(result, 'resets', 0, 1)

    def test_large_finite_loops_use_arithmetic(self):
        circuit = QuantumCircuit(1)
        with circuit.for_loop(range(10**12)):
            circuit.t(0)
        self.assertBound(costs(circuit), 't-count', 10**12, 10**12)
        empty = QuantumCircuit(1)
        with empty.for_loop(range(0)):
            empty.t(0)
        self.assertBound(costs(empty), 't-count', 0, 0)

    def test_break_and_continue_skip_unreachable_operations(self):
        circuit = QuantumCircuit(1)
        with circuit.for_loop(range(3)):
            circuit.t(0)
            circuit.continue_loop()
            circuit.t(0)
        with circuit.for_loop(range(8)):
            circuit.h(0)
            circuit.break_loop()
            circuit.t(0)
        self.assertBound(costs(circuit), 'gate-count', 4, 4)
        self.assertBound(costs(circuit), 't-count', 3, 3)

    def test_while_bounds_and_constant_conditions(self):
        for condition, bounds in ((False, (0, 0)), (True, (0, None)), (None, (0, None))):
            circuit = QuantumCircuit(1, 1)
            test = (circuit.clbits[0], True) if condition is None else expr.lift(condition)
            with circuit.while_loop(test):
                circuit.t(0)
            self.assertBound(costs(circuit), 't-count', *bounds)
        circuit = QuantumCircuit(1)
        with circuit.while_loop(expr.lift(True)):
            circuit.t(0)
            circuit.break_loop()
        self.assertBound(costs(circuit), 't-count', 1, 1)

    def test_literal_if_condition_and_switch_cases(self):
        circuit = QuantumCircuit(1, 2)
        with circuit.if_test(expr.lift(False)):
            circuit.t(0)
        zero = QuantumCircuit(1, 2)
        zero.t(0)
        other = QuantumCircuit(1, 2)
        other.h(0)
        other.h(0)
        circuit.switch(circuit.cregs[0], [(0, zero), (CASE_DEFAULT, other)], circuit.qubits, circuit.clbits)
        self.assertBound(costs(circuit), 'gate-count', 1, 2)
        self.assertBound(costs(circuit), 't-count', 0, 1)

    def test_unknown_gate_and_symbolic_rotation_have_unknown_t_upper_bound(self):
        for gate in ('ccx', 'rz'):
            circuit = QuantumCircuit(3, 1)
            with circuit.if_test((circuit.clbits[0], True)):
                if gate == 'ccx':
                    circuit.ccx(0, 1, 2)
                else:
                    circuit.rz(Parameter('a'), 0)
            self.assertBound(costs(circuit), 't-count', 0, None)
            self.assertBound(costs(circuit), 'gate-count', 0, 1)

    def test_open_pauli_controls_have_zero_t_cost(self):
        circuit = parse(HEADER + 'qubit[2] q; bit b; if(b) { negctrl @ x q[0],q[1]; }')
        self.assertBound(costs(circuit), 't-count', 0, 0)
        self.assertBound(costs(circuit), '2q-count', 0, 1)

    def test_controlled_clifford_phases_have_zero_t_cost_without_known_probabilities(self):
        for constructor in (CPhaseGate, CU1Gate):
            for angle in (-2 * math.pi, -math.pi, 0, math.pi, 2 * math.pi):
                for state in (0, 1):
                    with self.subTest(gate=constructor.__name__, angle=angle, control=state):
                        operation = constructor(angle, ctrl_state=state)
                        # Independently verify the recognized gate's unitary.
                        reference = QuantumCircuit(2)
                        if abs(angle) == math.pi:
                            if state == 0:
                                reference.x(0)
                            reference.append(CZGate(), [0, 1])
                            if state == 0:
                                reference.x(0)
                        self.assertTrue(Operator(operation).equiv(Operator(reference)))
                        circuit = QuantumCircuit(2, 1)
                        # b is uninitialized: this result must come from gate
                        # classification, not from the expected-cost analysis.
                        with circuit.if_test((circuit.clbits[0], True)):
                            circuit.append(operation, [0, 1])
                        result = costs(circuit)
                        self.assertBound(result, 't-count', 0, 0)
                        self.assertEqual(result['expected-counts']['t-count'],
                                         {'numerator': 0, 'denominator': 1})
                        self.assertIsNone(result['expected-counts']['gate-count'])

    def test_other_controlled_phases_keep_unknown_t_costs(self):
        for constructor in (CPhaseGate, CU1Gate):
            for angle in (math.pi / 2, math.nextafter(math.pi, math.inf), Parameter('theta')):
                for state in (0, 1):
                    with self.subTest(gate=constructor.__name__, angle=angle, control=state):
                        circuit = QuantumCircuit(3, 1)
                        circuit.h(0)
                        circuit.measure(0, 0)
                        with circuit.if_test((circuit.clbits[0], True)):
                            circuit.append(constructor(angle, ctrl_state=state), [1, 2])
                        result = costs(circuit)
                        self.assertBound(result, 't-count', 0, None)
                        self.assertIsNone(result['expected-counts']['t-count'])

    def test_custom_gate_named_cp_is_not_mistaken_for_a_clifford_phase(self):
        body = QuantumCircuit(2)
        body.cp(Parameter('theta') / 2, 0, 1)
        fake = body.to_gate()
        fake.name = 'cp'
        circuit = QuantumCircuit(2)
        circuit.append(fake, [0, 1])
        operation = circuit.assign_parameters({'theta': math.pi}).data[0].operation
        self.assertIsNone(_dynamic_t_cost(operation))

    def test_expected_counts_use_the_same_cost_callback_as_bounds(self):
        circuit = QuantumCircuit(3, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.cp(math.pi, 1, 2)
        result = collect_circuit_dynamic_costs(circuit, lambda op: 7 if op.name == 'cp' else 0)
        self.assertBound(result, 't-count', 0, 7)
        self.assertEqual(result['expected-counts']['t-count'], {'numerator': 7, 'denominator': 2})

    def test_reported_adders_have_fixed_t_costs(self):
        for name, count in (('adder-11.qasm', 12), ('adder-31.qasm', 36)):
            for flags in (['--json'], ['--clifford-t', '--json'], ['--clifford-t-ext', '--json'], ['--cost']):
                with self.subTest(source=name, flags=flags):
                    result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *flags,
                                             str(ROOT / 'tests/circuits' / name)],
                                            text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if '--json' in flags:
                        report = json.loads(result.stdout)
                        self.assertBound(report, 't-count', count, count)
                        self.assertEqual(report['expected-counts']['t-count'],
                                         {'numerator': count, 'denominator': 1})
                    else:
                        self.assertIn(f'  t-count: {count}\n', result.stdout)
                        self.assertNotIn(f't-count: {count}..unknown', result.stdout)
                        self.assertNotIn('t-count:', result.stdout.split('Expected execution counts')[1])

    def test_bounds_contain_enumerated_nested_paths(self):
        circuit = parse(HEADER + '''qubit[2] q; bit a; bit b;
for int i in [0:2] {
    if(a) { t q[0]; if(b) { t q[1]; } }
    else { h q[0]; h q[1]; }
}
''')
        result = costs(circuit)
        for a, b in itertools.product((False, True), repeat=2):
            gate_count = 3 * ((1 + int(b)) if a else 2)
            t_count = 3 * (int(a) * (1 + int(b)))
            for key, count in (('gate-count', gate_count), ('t-count', t_count)):
                self.assertLessEqual(result[key]['lower'], count)
                self.assertGreaterEqual(result[key]['upper'], count)

    def test_cli_default_qasm2_qasm3_and_json_cost(self):
        sources = [HEADER + 'qubit q; bit b; if(b) { t q; }',
                   'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg b[1]; if(b==1) t q[0];']
        for source in sources:
            for flags in (['--json'], ['--json', '--cost'], ['--cost']):
                result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *flags],
                                        input=source, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                if '--json' in flags:
                    self.assertBound(json.loads(result.stdout), 't-count', 0, 1)
                else:
                    self.assertIn('t-count: 0..1', result.stdout)
                    self.assertIn('Depth is not estimated', result.stdout)

    def test_mixed_static_and_dynamic_files_produce_one_json_array(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('static.qasm', 'dynamic.qasm')]
            paths[0].write_text(HEADER + 'qubit q; t q;')
            paths[1].write_text(HEADER + 'qubit q; bit b; if(b) { t q; }')
            result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), '--json', '--cost', *map(str, paths)],
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            static, dynamic = json.loads(result.stdout)
            self.assertEqual(static['t-count'], 1)
            self.assertBound(dynamic, 't-count', 0, 1)


if __name__ == '__main__':
    unittest.main()
