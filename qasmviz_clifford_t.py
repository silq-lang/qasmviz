"""Shared Clifford+T configuration (Qiskit 2.5 and later)."""

import math

from qiskit.transpiler import generate_preset_clifford_t_pass_manager


CLIFFORD_T_BASIS = ["cx", "h", "s", "sdg", "t", "tdg"]
DEFAULT_SYNTHESIS_ERROR = 1e-10


def validate_epsilon(epsilon):
    """Keep the historical zero-as-default spelling; reject invalid budgets."""
    if epsilon is not None and (not math.isfinite(epsilon) or not 0 <= epsilon <= 1):
        raise ValueError("--eps must be finite and between 0 and 1 (0 uses the default)")
    return DEFAULT_SYNTHESIS_ERROR if epsilon in (None, 0) else epsilon


def clifford_t_pass_manager(
    *, optimization_level=3, epsilon=None, hls_config=None, qubits_initially_zero=True
):
    """Build the dedicated pipeline with an explicit per-RZ error budget.

    Disable approximate cache matching so it cannot spend error in addition to
    the requested synthesis budget. This is not a whole-program error bound:
    several synthesized rotations, or repeated execution, accumulate error.
    """
    return generate_preset_clifford_t_pass_manager(
        optimization_level=optimization_level,
        basis_gates=CLIFFORD_T_BASIS,
        seed_transpiler=777,
        hls_config=hls_config,
        qubits_initially_zero=qubits_initially_zero,
        rz_synthesis_config={
            "rz_synthesis_error": validate_epsilon(epsilon),
            "rz_cache_error": 0.0,
        },
    )
