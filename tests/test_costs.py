import inspect
import math
import unittest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

from qasmviz import (
    _CLIFFORD_GATES, collect_costs, cx_metrics, metric_depth_and_count,
    rotation_metrics, t_cost_fully_known, t_metrics,
)


def costs(circuit, **targets):
    options = {name: False for name in inspect.signature(collect_costs).parameters
               if name != 'circuit'}
    options.update(targets)
    return collect_costs(circuit, **options)


class CostTests(unittest.TestCase):
    def test_barriers_do_not_add_t_layers(self):
        circuit = QuantumCircuit(1)
        circuit.barrier()
        circuit.t(0)
        circuit.barrier()
        circuit.barrier()
        circuit.tdg(0)
        circuit.barrier()
        self.assertEqual(t_metrics(circuit), (2, 2))
        result = costs(circuit, clifford_t=True)
        self.assertEqual((result['t-count'], result['t-depth']), (2, 2))

    def test_barriers_do_not_add_cx_layers(self):
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.cx(0, 1)
        self.assertEqual(cx_metrics(circuit), (2, 2))
        result = costs(circuit, cx_u=True)
        self.assertEqual((result['cx-count'], result['cx-depth']), (2, 2))

    def test_barriers_still_synchronize_only_their_operands(self):
        circuit = QuantumCircuit(3)
        circuit.t(0)
        circuit.t(0)
        circuit.barrier(0, 1)
        circuit.t(1)
        circuit.t(2)
        self.assertEqual(t_metrics(circuit), (4, 3))
        independent = QuantumCircuit(2)
        independent.t(0)
        independent.barrier(0)
        independent.t(1)
        self.assertEqual(t_metrics(independent), (2, 1))

    def test_weighted_rotation_depth(self):
        circuit = QuantumCircuit(1)
        circuit.rz(math.pi / 4, 0)
        circuit.barrier()
        circuit.rz(math.pi / 2, 0)
        circuit.barrier()
        circuit.rz(math.pi / 4, 0)
        self.assertEqual(rotation_metrics(circuit)[:4], (3, 3, 2, 0))

    def test_barriers_have_zero_weight_even_for_all_operations(self):
        circuit = QuantumCircuit(2)
        circuit.barrier()
        self.assertEqual(metric_depth_and_count(circuit, is_interesting=lambda _: True), (0, 0))

    def test_non_clifford_controls_do_not_claim_exact_rotation_t_depth(self):
        for name in ('ch', 'cs', 'csdg'):
            with self.subTest(gate=name):
                circuit = QuantumCircuit(2)
                getattr(circuit, name)(0, 1)
                circuit.rz(math.pi / 4, 0)
                self.assertFalse(t_cost_fully_known(circuit))
                self.assertNotIn('t-depth', costs(circuit)['rotations'])

    def test_clifford_classification_is_valid(self):
        # Verify the mathematical property independently of the name table.
        from qiskit.circuit.library import get_standard_gate_name_mapping
        gates = get_standard_gate_name_mapping()
        for name in _CLIFFORD_GATES:
            with self.subTest(gate=name):
                gate = gates[name]
                circuit = QuantumCircuit(gate.num_qubits)
                circuit.append(gate, range(gate.num_qubits))
                Clifford(circuit)
                self.assertTrue(t_cost_fully_known(circuit))


if __name__ == '__main__':
    unittest.main()
