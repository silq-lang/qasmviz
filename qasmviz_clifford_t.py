"""Shared Clifford+T configuration (Qiskit 2.5 and later)."""

import math

from qiskit.transpiler import PassManager, generate_preset_clifford_t_pass_manager

from qasmviz_basis import (
    CliffordBasisTranslation, EXTENDED_CLIFFORD_T_BASIS, validate_clifford_basis,
)


CLIFFORD_T_BASIS = ["cx", "h", "s", "sdg", "t", "tdg"]
DEFAULT_SYNTHESIS_ERROR = 1e-10


def validate_epsilon(epsilon):
    """Keep the historical zero-as-default spelling; reject invalid budgets."""
    if epsilon is not None and (not math.isfinite(epsilon) or not 0 <= epsilon <= 1):
        raise ValueError("--eps must be finite and between 0 and 1 (0 uses the default)")
    return DEFAULT_SYNTHESIS_ERROR if epsilon in (None, 0) else epsilon


def clifford_t_pass_manager(
    *, optimization_level=3, epsilon=None, hls_config=None, qubits_initially_zero=True,
    basis_gates=None,
):
    """Build the dedicated pipeline with an explicit per-RZ error budget.

    Disable approximate cache matching so it cannot spend error in addition to
    the requested synthesis budget. This is not a whole-program error bound:
    several synthesized rotations, or repeated execution, accumulate error.
    """
    basis = None if basis_gates is None else validate_clifford_basis(basis_gates)
    # Synthesizing directly to an unusual Clifford basis can make Qiskit's
    # translator use T gates for missing Cliffords. Keep synthesis in the full
    # named basis, then translate Cliffords exactly into the requested set.
    pm = generate_preset_clifford_t_pass_manager(
        optimization_level=optimization_level,
        basis_gates=CLIFFORD_T_BASIS if basis is None else list(EXTENDED_CLIFFORD_T_BASIS),
        seed_transpiler=777,
        hls_config=hls_config,
        qubits_initially_zero=qubits_initially_zero,
        rz_synthesis_config={
            "rz_synthesis_error": validate_epsilon(epsilon),
            "rz_cache_error": 0.0,
        },
    )
    if basis is not None:
        pm.post_scheduling = PassManager([CliffordBasisTranslation(basis)])
    return pm
