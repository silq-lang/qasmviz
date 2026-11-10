import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qasmviz_clifford_t import (
    CLIFFORD_T_BASIS,
    DEFAULT_SYNTHESIS_ERROR,
    clifford_t_pass_manager,
    validate_epsilon,
)


class CliffordTTests(unittest.TestCase):
    def test_error_budget(self):
        original = QuantumCircuit(1)
        original.rz(0.123456789, 0)
        for epsilon in (1e-3, 1e-6):
            with self.subTest(epsilon=epsilon):
                compiled = clifford_t_pass_manager(
                    optimization_level=1, epsilon=epsilon, qubits_initially_zero=False
                ).run(original)
                self.assertLessEqual(set(compiled.count_ops()), set(CLIFFORD_T_BASIS))
                error = np.linalg.norm(Operator(original).data - Operator(compiled).data, 2)
                self.assertLessEqual(error, epsilon + 1e-12)

    def test_exact_gates_need_no_approximation(self):
        original = QuantumCircuit(2)
        original.x(0)
        original.h(1)
        original.cx(0, 1)
        original.t(0)
        compiled = clifford_t_pass_manager(
            optimization_level=1, qubits_initially_zero=False
        ).run(original)
        np.testing.assert_allclose(Operator(original).data, Operator(compiled).data, atol=1e-12)
        self.assertLessEqual(set(compiled.count_ops()), set(CLIFFORD_T_BASIS))

    def test_zero_retains_default_semantics(self):
        self.assertEqual(validate_epsilon(0), DEFAULT_SYNTHESIS_ERROR)
        self.assertEqual(validate_epsilon(None), DEFAULT_SYNTHESIS_ERROR)

    def test_invalid_budgets(self):
        for value in (float("nan"), float("inf"), -float("inf"), -1e-6, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_epsilon(value)


if __name__ == "__main__":
    unittest.main()
