import warnings

warnings.filterwarnings(
    "ignore",
    message=r"(?s).*You have imported samplomatic==.*beta development.*",
    category=UserWarning,
    module=r"samplomatic(\..*)?",
)

import sys
import re
import math
import cmath
import argparse

from qiskit_qasm3_import import parse
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import qblaze
import qblaze.qiskit


CLIFFORD_T_BASIS = ["cx", "h", "s", "sdg", "t", "tdg"]
CX_U_BASIS = ["cx", "u"]
CX_SX_BASIS = ["rz", "sx", "cx"]
ECR_SX_BASIS = ["rz", "sx", "ecr"]
CZ_SX_BASIS = ["rz", "sx", "cz"]
ISWAP_RX_BASIS = ["rz", "rx", "iswap"]
RZZ_RX_BASIS = ["rz", "rx", "rzz"]
RXX_RX_BASIS = ["rz", "rx", "rxx"]

# Vendor-exact bases
IBM_EAGLE_BASIS = ["ecr", "id", "rz", "sx", "x"]
IBM_HERON_BASIS = ["cz", "id", "rz", "sx", "x"]
IBM_HERON_FRAC_BASIS = ["cz", "id", "rz", "sx", "x", "rx", "rzz"]
RIGETTI_ANKAA_BASIS = ["rx", "rz", "iswap"]


def arg_norm(x: float) -> float:
    while x <= -math.pi:
        x += 2 * math.pi
    while x > math.pi:
        x -= 2 * math.pi
    return x


def arg_close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(arg_norm(a - b)) < tol


def print_table(rows: list[tuple[str, str, str]]) -> None:
    if not rows:
        return
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    for a, b, c in rows:
        print(f"{a.rjust(widths[0])} {b.ljust(widths[1])}{c}")


def print_pretty_state(
    values: list[tuple[str, complex]],
    max_abs: float,
    error_prob: float = 0.0,
    show_error_prob: bool = False,
) -> None:
    if not values:
        print("0")
        if show_error_prob:
            print(f"Error probability: {error_prob}")
        return

    arg0 = cmath.phase(values[0][1])
    arg1 = arg_norm(arg0 + math.pi)
    if all(
        arg_close(arg0, cmath.phase(v)) or arg_close(arg1, cmath.phase(v))
        for _, v in values
    ):
        any_arg = False
    else:
        arg0 = 0.0
        any_arg = True

    exp = math.floor(-math.log10(max_abs**2)) if max_abs > 0 else 0
    if exp > 0:
        exp_suffix = f"e-{exp}"
        exp_scale = 10**exp
    else:
        exp_suffix = ""
        exp_scale = 1

    rows: list[tuple[str, str, str]] = []
    is_first = True

    for basis, val in values:
        val_abs, val_arg = cmath.polar(val)
        val_arg = arg_norm(val_arg - arg0)

        if val_arg > math.pi / 2:
            val_sign = True
            val_arg -= math.pi
        elif val_arg <= -math.pi / 2:
            val_sign = True
            val_arg += math.pi
        else:
            val_sign = False

        if not any_arg:
            assert abs(val_arg) < 1e-5

        rows.append((
            f'{"-" if val_sign else " " if is_first else "+"}√{val_abs**2 * exp_scale:.07f}{exp_suffix}',
            f"∠{val_arg:.7f}" if any_arg else "",
            f" · |{basis}⟩",
        ))
        is_first = False

    print_table(rows)
    if show_error_prob:
        print(f"Error probability: {error_prob}")


def format_clbits(clbits) -> str:
    def clbit_sort_key(bit) -> tuple[int, str]:
        idx = getattr(bit, "_index", None)
        if idx is None:
            idx = 0
        return (idx, repr(bit))

    if clbits is None:
        return ""

    if hasattr(clbits, "items"):
        items = sorted(
            clbits.items(),
            key=lambda kv: clbit_sort_key(kv[0]),
        )
        return "".join("1" if int(v) else "0" for _, v in items)

    items = list(clbits)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        items = sorted(items, key=lambda kv: clbit_sort_key(kv[0]))
        return "".join("1" if int(v) else "0" for _, v in items)

    return "".join("1" if int(v) else "0" for v in items)


def sparse_statevector_from_sim(sim, nqubits: int) -> list[tuple[str, complex]]:
    values: list[tuple[int, complex]] = []

    for index, amp in sim:
        amp = complex(amp)
        if abs(amp) > 1e-15:
            values.append((int(index), amp))

    values.sort(key=lambda kv: kv[0])

    return [
        (format(index, f"0{nqubits}b"), amp)
        for index, amp in values
    ]


def run_circuit_and_capture(circuit) -> tuple[object, list[tuple[str, complex]]]:
    sim = qblaze.Simulator()
    clbits, _statevectors = qblaze.qiskit.run_circuit(sim, circuit)
    values = sparse_statevector_from_sim(sim, circuit.num_qubits)
    return clbits, values


from collections.abc import Callable

from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode


def metric_depth_and_count(
    circuit,
    *,
    is_interesting: Callable[[DAGOpNode], bool],
    node_weight: Callable[[DAGOpNode], int] | None = None,
    respect_barriers: bool = True,
    dag=None,
) -> tuple[int, int]:
    """
    Dependency-aware metric depth/count on the circuit DAG.

    The metric depth is the length of the longest dependency-respecting path,
    counting:
      - ``node_weight(node)`` for each interesting node (default: 1)
      - additionally 1 for each barrier node if respect_barriers=True

    Non-interesting, non-barrier nodes propagate the current metric layer
    without increasing it.

    An already-built DAG may be supplied via ``dag`` to avoid rebuilding it.
    """
    if dag is None:
        dag = circuit_to_dag(circuit)

    level: dict[int, int] = {}
    count = 0
    depth = 0

    for node in dag.topological_op_nodes():
        preds = [
            pred
            for pred in dag.predecessors(node)
            if isinstance(pred, DAGOpNode)
        ]
        base = max((level[pred._node_id] for pred in preds), default=0)

        is_barrier = respect_barriers and node.op.name == "barrier"
        interesting = is_interesting(node)

        weight = (node_weight(node) if node_weight else 1) if interesting else 0
        here = base + (weight if interesting else (1 if is_barrier else 0))
        level[node._node_id] = here

        if interesting:
            count += 1
            if here > depth:
                depth = here

    return depth, count

def t_metrics(circuit) -> tuple[int, int]:
    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=lambda node: node.op.name in {"t", "tdg"},
        respect_barriers=True,
    )
    return count, depth


def cx_metrics(circuit) -> tuple[int, int]:
    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=lambda node: node.op.name == "cx",
        respect_barriers=True,
    )
    return count, depth


# Gates that are not themselves interesting as "two-qubit entangling" ops for
# cost purposes: classical bookkeeping, barriers, and measurements.
_NON_GATE_OPS = {"barrier", "measure", "reset", "delay", "snapshot"}


def two_qubit_gate_names(circuit) -> set[str]:
    """Return the set of distinct two-qubit gate names present in the circuit."""
    names: set[str] = set()
    for instr in circuit.data:
        op = instr.operation
        if op.name not in _NON_GATE_OPS and op.num_qubits == 2:
            names.add(op.name)
    return names


def has_many_qubit_gates(circuit) -> bool:
    """Return True if the circuit contains any gate acting on 3 or more qubits."""
    return any(
        instr.operation.num_qubits >= 3
        for instr in circuit.data
        if instr.operation.name not in _NON_GATE_OPS
    )


def ecr_metrics(circuit) -> tuple[int, int]:
    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=lambda node: node.op.name == "ecr",
        respect_barriers=True,
    )
    return count, depth


def sx_metrics(circuit) -> tuple[int, int]:
    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=lambda node: node.op.name in {"sx", "sxdg"},
        respect_barriers=True,
    )
    return count, depth


def two_qubit_metrics(circuit) -> tuple[int, int]:
    """Return (count, depth) for all two-qubit gates."""
    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=lambda node: (
            node.op.name not in _NON_GATE_OPS and node.op.num_qubits == 2
        ),
        respect_barriers=True,
    )
    return count, depth


# Gates tracked for rotation cost analysis: parameterized rotation gates plus
# fixed-angle single-qubit non-Clifford/Clifford gates whose T-cost is known.
_ROTATION_GATES = {"rx", "ry", "rz", "p", "u1", "u2", "u3", "u", "s", "sdg", "t", "tdg"}

# Fixed T-costs for named gates that have no angle parameters.
_FIXED_T_COST: dict[str, int] = {
    "t": 1, "tdg": 1,
    "s": 0, "sdg": 0,
}

# Standard named gates with known zero T-cost.  Does not include rotation
# gates (whose T-cost depends on the angle) or T/Tdg (T-cost 1).
_CLIFFORD_GATES = {
    # Single-qubit
    "h", "s", "sdg", "x", "y", "z", "sx", "sxdg", "id",
    # Two-qubit
    "cx", "cy", "cz", "ch", "cs", "csdg", "swap", "iswap", "dcx", "ecr",
}


def _dyadic_t_cost(angle: float) -> int | None:
    """
    Return the exact T-count required to synthesize a rotation by ``angle``
    radians, or None if exact ancilla-free Clifford+T synthesis is not possible.

    A rotation by kπ/2^n (in lowest terms, k odd, n ≥ 0):
      - n=0: multiple of π      → identity or Z/X up to Clifford → 0
      - n=1: multiple of π/2    → S/Sdg or X/Y up to Clifford   → 0
      - n=2: multiple of π/4    → T/Tdg up to Clifford           → 1
      - n>2: not exactly representable in the Clifford+T group    → None

    Returns None for n > 2 and for angles that are not dyadic rational
    multiples of π; both cases require approximate synthesis whose cost
    depends on the target precision ε.
    """
    # Normalise to [0, 2π).
    turns = (angle / math.pi) % 2.0   # angle in units of π, mod 2

    # Check if turns is close to a dyadic rational k/2^n for increasing n.
    MAX_N = 2
    for n in range(MAX_N + 1):
        denom = 2 ** n
        k = round(turns * denom)
        if abs(turns - k / denom) < 1e-9:
            # Reduce k/2^n to lowest terms by cancelling common factors of 2.
            while n > 0 and k % 2 == 0:
                k //= 2
                n -= 1
            # n is now the 2-adic order of the denominator in lowest terms.
            # Clifford gates correspond to n ≤ 1 (multiples of π/2).
            # n == 2 (multiples of π/4) cost exactly 1 T-gate.
            # n > 2 cannot be exactly synthesized in ancilla-free Clifford+T;
            # they require approximate synthesis whose cost depends on ε.
            if n <= 1:
                return 0
            if n == 2:
                return 1
            return None
    return None  # not a dyadic rational multiple of π


def _gate_t_cost(op) -> int | None:
    """
    Return the T-cost of a rotation gate operation.

    For fixed-angle named gates (s, sdg, t, tdg), returns the known cost
    directly.  For parameterized gates (rz, u, etc.), computes cost from
    the angle.  For gates with multiple angle parameters (u, u2, u3), the
    cost is the maximum over all parameters.

    Returns None if any parameter requires approximate synthesis.
    """
    if op.name in _FIXED_T_COST:
        return _FIXED_T_COST[op.name]
    costs = []
    for param in op.params:
        try:
            angle = float(param)
        except (TypeError, Exception):
            return None  # symbolic parameter — can't determine statically
        c = _dyadic_t_cost(angle)
        if c is None:
            return None
        costs.append(c)
    return sum(costs)


def rotation_metrics(circuit) -> tuple[int, int, int, int, dict[int | None, int]]:
    """
    Return (count, depth, t_depth, n_approx, breakdown) for arbitrary-angle
    single-qubit rotation gates.

    ``t_depth`` is the weighted depth counting only the dyadic-rotation
    contribution (each gate weighted by its exact T-cost).  Approximate
    rotations contribute 0 to this number.

    ``n_approx`` is the number of approximate rotations (non-dyadic angles).
    The full T-depth is ``t_depth + n_approx·(synthesis cost per gate)``;
    since the per-gate cost depends on the target precision, we report the
    two parts separately.

    ``breakdown`` maps T-cost → number of rotation gates with that cost:
      - 0      : Clifford rotations (angle is a multiple of π/2)
      - 1      : T-gates (π/4)
      - n > 1  : dyadic rotations requiring n T-gates (angle kπ/2^n)
      - None   : approximate rotations (non-dyadic angle)
    """
    breakdown: dict[int | None, int] = {}
    t_costs: dict[int, int] = {}   # node_id -> T-cost (approx nodes absent)

    def is_rotation_collect(node: DAGOpNode) -> bool:
        if node.op.name not in _ROTATION_GATES:
            return False
        cost = _gate_t_cost(node.op)
        breakdown[cost] = breakdown.get(cost, 0) + 1
        if cost is not None:
            t_costs[node._node_id] = cost
        return True

    def is_rotation(node: DAGOpNode) -> bool:
        return node.op.name in _ROTATION_GATES

    dag = circuit_to_dag(circuit)

    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=is_rotation_collect,
        respect_barriers=True,
        dag=dag,
    )

    n_approx = breakdown.get(None, 0)

    if n_approx == 0:
        def t_weight(node: DAGOpNode) -> int:
            return t_costs.get(node._node_id, 0)

        t_depth_val, _ = metric_depth_and_count(
            circuit,
            is_interesting=is_rotation,
            node_weight=t_weight,
            respect_barriers=True,
            dag=dag,
        )
    else:
        t_depth_val = 0  # not meaningful; caller checks n_approx

    return count, depth, t_depth_val, n_approx, breakdown



def t_cost_fully_known(circuit) -> bool:
    """
    Return True if every gate in the circuit has a known T-cost, meaning
    T-depth can be computed exactly from the rotation gate analysis.

    A gate has known T-cost if it is:
      - in _CLIFFORD_GATES (cost 0)
      - t or tdg (cost 1)
      - in _ROTATION_GATES (cost determined by angle — may still be approx)
      - in _NON_GATE_OPS (measurements, barriers, resets — not gate costs)

    Any other gate name is conservatively assumed to have unknown T-cost
    (e.g. CCX, custom gates not in the standard sets).
    """
    _known = _CLIFFORD_GATES | {"t", "tdg"} | _ROTATION_GATES | _NON_GATE_OPS
    return all(
        instr.operation.name in _known
        for instr in circuit.data
    )

def _iter_condition_clbits(cond):
    """
    Yield Clbit objects referenced by a Qiskit control-flow condition/target.

    Supports the common forms:
      - (Clbit, int)
      - (ClassicalRegister, int)
      - Clbit
      - ClassicalRegister
      - expr.Expr (via conservative recursive introspection)
    """
    from qiskit.circuit import Clbit, ClassicalRegister

    seen = set()

    def visit(obj):
        if obj is None:
            return

        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, Clbit):
            yield obj
            return

        if isinstance(obj, ClassicalRegister):
            for bit in obj:
                yield bit
            return

        if isinstance(obj, tuple) and len(obj) == 2:
            lhs, _rhs = obj
            yield from visit(lhs)
            return

        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                yield from visit(item)
            return

        # Conservative expr-like traversal. Qiskit's runtime classical
        # expressions are tree-structured; these names cover the common cases.
        for attr in ("left", "right", "operand", "operands", "children", "target"):
            if hasattr(obj, attr):
                try:
                    child = getattr(obj, attr)
                except Exception:
                    continue
                yield from visit(child)

    yield from visit(cond)


def _block_has_quantum_effects(block) -> bool:
    """
    Return True if the block contains any quantum-affecting operation, including
    nested control flow whose blocks have quantum effects.
    """
    from qiskit.circuit import ControlFlowOp

    for instruction in block.data:
        op = instruction.operation

        # Ignore purely classical terminal-ish work.
        if op.name in {"measure", "barrier"}:
            continue

        # Any op touching qubits is quantum-relevant.
        if instruction.qubits:
            return True

        # Recurse into nested control flow.
        if isinstance(op, ControlFlowOp):
            for inner in op.blocks:
                if _block_has_quantum_effects(inner):
                    return True

    return False


def active_measurement_indices(circuit) -> set[int]:
    """
    Return top-level instruction indices of measurements whose results
    influence later quantum execution through control flow.

    Semantics:
      - only measurements that write clbits later consumed by If/While/Switch
        conditions that gate quantum-affecting blocks are considered
      - a later overwrite of a clbit kills the earlier measurement's activity
      - nested measurements are not returned (only top-level indices)
    """
    from qiskit.circuit import ControlFlowOp, IfElseOp, WhileLoopOp, SwitchCaseOp

    data = list(circuit.data)
    active: set[int] = set()
    # clbit -> index of last top-level measure writing it, or -1 if overwritten
    # by a nested measure (so no longer attributable to a top-level instruction)
    current_defs: dict[object, int] = {}

    def process_block(block, reaching_defs: dict) -> None:
        local = dict(reaching_defs)
        for instr in block.data:
            op = instr.operation
            if op.name == "measure":
                for cbit in instr.clbits:
                    local[cbit] = -1
                continue
            if isinstance(op, (IfElseOp, WhileLoopOp)):
                used = set(_iter_condition_clbits(op.condition))
                if _block_has_quantum_effects(block) and used:
                    for cbit in used:
                        src = local.get(cbit)
                        if src is not None and src >= 0:
                            active.add(src)
                for inner in op.blocks:
                    process_block(inner, local)
                continue
            if isinstance(op, SwitchCaseOp):
                used = set(_iter_condition_clbits(op.target))
                if _block_has_quantum_effects(block) and used:
                    for cbit in used:
                        src = local.get(cbit)
                        if src is not None and src >= 0:
                            active.add(src)
                for inner in op.blocks:
                    process_block(inner, local)
                continue
            if isinstance(op, ControlFlowOp):
                for inner in op.blocks:
                    process_block(inner, local)

    for i, instr in enumerate(data):
        op = instr.operation
        if op.name == "measure":
            for cbit in instr.clbits:
                current_defs[cbit] = i
            continue
        if isinstance(op, (IfElseOp, WhileLoopOp)):
            used = set(_iter_condition_clbits(op.condition))
            if any(_block_has_quantum_effects(inner) for inner in op.blocks):
                for cbit in used:
                    src = current_defs.get(cbit)
                    if src is not None and src >= 0:
                        active.add(src)
            for inner in op.blocks:
                process_block(inner, current_defs)
            continue
        if isinstance(op, SwitchCaseOp):
            used = set(_iter_condition_clbits(op.target))
            if any(_block_has_quantum_effects(inner) for inner in op.blocks):
                for cbit in used:
                    src = current_defs.get(cbit)
                    if src is not None and src >= 0:
                        active.add(src)
            for inner in op.blocks:
                process_block(inner, current_defs)
            continue
        if isinstance(op, ControlFlowOp):
            for inner in op.blocks:
                process_block(inner, current_defs)

    return active


def mcm_metrics(circuit) -> tuple[int, int]:
    """
    Return (count, depth) for active mid-circuit measurements.

    Uses the same DAG-based dependency tracking as t_metrics and cx_metrics:
    metric_depth_and_count walks the circuit DAG in topological order, so both
    qubit-wire and clbit-wire data dependencies are automatically respected.

    Active measurements are identified by the clbit-flow analysis in
    active_measurement_indices.  Each active instruction index is mapped to its
    corresponding DAGOpNode by pairing circuit.data and dag.topological_op_nodes()
    in order, filtering both to measure ops.  This is correct because
    circuit_to_dag processes circuit.data sequentially, so measure nodes appear
    in the same relative order in both sequences.
    """
    active_instr_indices = active_measurement_indices(circuit)
    if not active_instr_indices:
        return 0, 0

    # Map circuit instruction indices → DAG node IDs by walking the DAG in
    # topological order and matching measure ops in the order they appear.
    # circuit_to_dag processes circuit.data in order, so measure nodes appear
    # in the same relative order in both circuit.data and the DAG's topological
    # traversal.  We pair them up by counting measure ops seen so far.
    dag = circuit_to_dag(circuit)
    measure_instr_indices = [
        i for i, instr in enumerate(circuit.data)
        if instr.operation.name == "measure"
    ]
    active_node_ids: set[int] = set()
    measure_dag_nodes = [
        node for node in dag.topological_op_nodes()
        if node.op.name == "measure"
    ]
    assert len(measure_dag_nodes) == len(measure_instr_indices), (
        "DAG measure node count doesn't match circuit.data measure count"
    )
    for instr_idx, dag_node in zip(measure_instr_indices, measure_dag_nodes):
        if instr_idx in active_instr_indices:
            active_node_ids.add(dag_node._node_id)

    def is_active_measure(node: DAGOpNode) -> bool:
        return node._node_id in active_node_ids

    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=is_active_measure,
        respect_barriers=True,
        dag=dag,
    )
    return count, depth


def mcm_count(circuit) -> int:
    count, _depth = mcm_metrics(circuit)
    return count


def mcm_depth(circuit) -> int:
    """
    Dependency-aware MCM depth: the number of layers of active mid-circuit
    measurements accounting for data dependencies, computed over the circuit
    DAG exactly as T-depth and CX-depth are.
    """
    _count, depth = mcm_metrics(circuit)
    return depth

def format_gate_counts(circuit, *, physical: bool = False) -> tuple[int, str]:
    """
    Return (display_total, breakdown_str).

    In normal mode: total is all gates, breakdown groups ops/measures/barriers
    at the back separated by ';'.

    In physical mode: display_total is physical-only count shown as 'N[+V]'
    where V is virtual gate count, and breakdown has three sections:
    'physical; reset/measure; virtual'.
    """
    counts = dict(circuit.count_ops())

    _MEASURE_OPS = {"reset", "measure", "barrier"}
    _VIRTUAL_GATE_NAMES = {"rz", "p", "u1", "id"} if physical else set()

    preferred_order = [
        # 2Q gates
        "cx", "ecr", "cz", "iswap", "rzz", "xxphase", "zzphase", "zzmax", "syc", "sqrt_iswap", "sqrt_iswap_inv",
        "swap", "xx", "yy", "zz", "rxx", "ryy",
        # fixed 1Q
        "h", "s", "sdg", "t", "tdg", "x", "y", "z", "sx", "sxdg",
        # parameterized 1Q
        "rx", "ry", "rz", "p", "phxz", "phasedx", "u", "u1", "u2", "u3", "id",
    ]
    rank = {name: i for i, name in enumerate(preferred_order)}

    def fmt(items):
        return ", ".join(f"{name}={count}" for name, count in items)

    def sort_key(kv):
        return (rank.get(kv[0], len(preferred_order)), kv[0])

    gate_items = sorted(
        [(k, v) for k, v in counts.items()
         if k not in _MEASURE_OPS and k not in _VIRTUAL_GATE_NAMES],
        key=sort_key,
    )
    measure_items = sorted(
        [(k, v) for k, v in counts.items() if k in _MEASURE_OPS],
        key=sort_key,
    )
    virtual_items = sorted(
        [(k, v) for k, v in counts.items() if k in _VIRTUAL_GATE_NAMES],
        key=sort_key,
    )

    sections = [fmt(gate_items)]
    if measure_items:
        sections.append(fmt(measure_items))
    if virtual_items:
        sections.append(fmt(virtual_items))
    breakdown = "; ".join(s for s in sections if s)

    physical_count = sum(v for k, v in counts.items()
                         if k not in _MEASURE_OPS and k not in _VIRTUAL_GATE_NAMES)
    virtual_count = sum(v for k, v in virtual_items)
    total = sum(counts.values())

    if physical and virtual_count:
        display_total = f"{physical_count}[+{virtual_count}]"
    else:
        display_total = str(total)

    return display_total, breakdown

def collect_costs(circuit, *, clifford_t: bool, cx_u: bool, cx_sx: bool, ecr_sx: bool, cz_sx: bool, iswap_rx: bool, rzz_rx: bool, rxx_rx: bool, xxphase_rx: bool, quantinuum_h: bool, ionq_aria: bool, ionq_forte: bool, syc_phxz: bool, sqrtiswap_phxz: bool, fez: bool, ibm_eagle: bool, ibm_heron: bool, ibm_heron_frac: bool, rigetti_ankaa: bool) -> dict:
    """
    Compute all cost metrics for the circuit and return them as a plain dict.
    This is the single source of truth consumed by both print_costs and
    print_costs_json.

    Keys present depend on the circuit; absent metrics are not included.
    Structured sub-fields (e.g. rotation breakdown) are nested dicts.
    """
    # rz is a virtual gate (frame change) on both superconducting and trapped-ion
    # hardware — it maps to a classical phase update with no pulse cost.
    virtual_rz = cx_sx or ecr_sx or cz_sx or iswap_rx or rzz_rx or rxx_rx or xxphase_rx or quantinuum_h or ionq_aria or ionq_forte or syc_phxz or sqrtiswap_phxz or fez or ibm_eagle or ibm_heron or ibm_heron_frac or rigetti_ankaa
    primitive_1q = "sx" if (cx_sx or ecr_sx or cz_sx or ibm_eagle or ibm_heron) else "rx" if (iswap_rx or rzz_rx or rxx_rx or xxphase_rx or rigetti_ankaa or ibm_heron_frac) else "phxz" if (syc_phxz or sqrtiswap_phxz) else "phasedx" if quantinuum_h else "gpi2" if (ionq_aria or ionq_forte) else None

    data: dict = {}

    data["width"] = circuit.num_qubits

    if virtual_rz:
        # rz is a virtual gate (frame change, zero pulse cost) and should
        # not contribute to circuit depth.
        _VIRTUAL_GATES = {"rz", "p", "u1", "id"}
        data["depth"] = circuit.depth(
            filter_function=lambda instr: instr.operation.name not in _VIRTUAL_GATES
        )
    else:
        data["depth"] = circuit.depth()

    gate_display_total, gate_breakdown = format_gate_counts(circuit, physical=virtual_rz)
    gate_counts = dict(circuit.count_ops())
    _MEASURE_OPS_SET = {"reset", "measure", "barrier"}
    _VIRTUAL_SET = {"rz", "p", "u1", "id"} if virtual_rz else set()

    def _sorted_gate_dict(predicate):
        preferred = [
            "cx", "ecr", "cz", "swap", "h", "s", "sdg", "t", "tdg",
            "x", "y", "z", "sx", "sxdg", "rx", "ry", "rz", "p",
            "u", "u1", "u2", "u3", "id",
        ]
        rank = {name: i for i, name in enumerate(preferred)}
        items = sorted(
            [(k, v) for k, v in gate_counts.items() if predicate(k)],
            key=lambda kv: (rank.get(kv[0], len(preferred)), kv[0]),
        )
        return dict(items)

    gates_dict = _sorted_gate_dict(
        lambda k: k not in _MEASURE_OPS_SET and k not in _VIRTUAL_SET
    )
    measure_dict = _sorted_gate_dict(lambda k: k in _MEASURE_OPS_SET)
    virtual_dict = _sorted_gate_dict(lambda k: k in _VIRTUAL_SET)

    total_count = sum(gate_counts.values())
    physical_count = sum(gates_dict.values())
    virtual_count = sum(virtual_dict.values())

    data["gates"] = {
        "display_count": gate_display_total,
        "total_count": total_count,
        "breakdown": gate_breakdown,
        "gates": gates_dict,
        "ops": measure_dict,
    }
    if virtual_rz:
        data["gates"]["physical_count"] = physical_count
        data["gates"]["virtual_count"] = virtual_count
        data["gates"]["virtual"] = virtual_dict

    if circuit.num_clbits:
        data["clbits"] = circuit.num_clbits

    twoq_names = two_qubit_gate_names(circuit)
    if clifford_t or cx_u or cx_sx:
        cxc, cxd = cx_metrics(circuit)
        if cxc:
            data["cx-count"] = cxc
            data["cx-depth"] = cxd
    elif ecr_sx:
        ecrc, ecrd = ecr_metrics(circuit)
        if ecrc:
            data["ecr-count"] = ecrc
            data["ecr-depth"] = ecrd
    elif cz_sx:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "cz",
            respect_barriers=True,
        )
        if count:
            data["cz-count"] = count
            data["cz-depth"] = depth
    elif iswap_rx:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "iswap",
            respect_barriers=True,
        )
        if count:
            data["iswap-count"] = count
            data["iswap-depth"] = depth
    elif rzz_rx:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "rzz",
            respect_barriers=True,
        )
        if count:
            data["rzz-count"] = count
            data["rzz-depth"] = depth
    elif rxx_rx:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "rxx",
            respect_barriers=True,
        )
        if count:
            data["rxx-count"] = count
            data["rxx-depth"] = depth
    elif xxphase_rx:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "xxphase",
            respect_barriers=True,
        )
        if count:
            data["xxphase-count"] = count
            data["xxphase-depth"] = depth
    elif quantinuum_h:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "zzphase",
            respect_barriers=True,
        )
        if count:
            data["zzphase-count"] = count
            data["zzphase-depth"] = depth
    elif ionq_aria:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "ms",
            respect_barriers=True,
        )
        if count:
            data["ms-count"] = count
            data["ms-depth"] = depth
    elif ionq_forte:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "zz",
            respect_barriers=True,
        )
        if count:
            data["zz-count"] = count
            data["zz-depth"] = depth
    elif syc_phxz:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "syc",
            respect_barriers=True,
        )
        if count:
            data["syc-count"] = count
            data["syc-depth"] = depth
    elif sqrtiswap_phxz:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name in {"sqrt_iswap", "sqrt_iswap_inv"},
            respect_barriers=True,
        )
        if count:
            data["sqrt_iswap-count"] = count
            data["sqrt_iswap-depth"] = depth
    elif ibm_eagle:
        ecrc, ecrd = ecr_metrics(circuit)
        if ecrc:
            data["ecr-count"] = ecrc
            data["ecr-depth"] = ecrd
    elif ibm_heron or ibm_heron_frac:
        if ibm_heron_frac:
            depth, count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name in {"cz", "rzz"},
                respect_barriers=True,
            )
            if count:
                data["2q-count"] = count
                data["2q-depth"] = depth
        else:
            depth, count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name == "cz",
                respect_barriers=True,
            )
            if count:
                data["cz-count"] = count
                data["cz-depth"] = depth
    elif rigetti_ankaa:
        depth, count = metric_depth_and_count(
            circuit,
            is_interesting=lambda node: node.op.name == "iswap",
            respect_barriers=True,
        )
        if count:
            data["iswap-count"] = count
            data["iswap-depth"] = depth
    elif has_many_qubit_gates(circuit):
        pass
    elif len(twoq_names) == 1:
        name = next(iter(twoq_names))
        if name == "cx":
            tqc, tqd = cx_metrics(circuit)
            key = "cx"
        elif name == "ecr":
            tqc, tqd = ecr_metrics(circuit)
            key = "ecr"
        else:
            depth, count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name == name,
                respect_barriers=True,
            )
            tqc, tqd = count, depth
            key = name
        if tqc:
            data[f"{key}-count"] = tqc
            data[f"{key}-depth"] = tqd
    elif twoq_names:
        tqc, tqd = two_qubit_metrics(circuit)
        if tqc:
            data["2q-count"] = tqc
            data["2q-depth"] = tqd

    rc, rd, rt, rn_approx, rbreakdown = rotation_metrics(circuit)
    _NAMED_ONLY = {"s", "sdg", "t", "tdg"}
    has_parametric = any(
        instr.operation.name in _ROTATION_GATES and instr.operation.name not in _NAMED_ONLY
        for instr in circuit.data
    )

    if primitive_1q is not None:
        if primitive_1q == "gpi2":
            # IonQ native: count both GPI2 (π/2 pulse) and GPI (π pulse) as 1Q primitives,
            # but report them together under "gpi2" to distinguish from the 2Q entangling gates.
            p1_depth, p1_count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name in {"gpi2", "gpi"},
                respect_barriers=True,
            )
        else:
            p1_depth, p1_count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node, _g=primitive_1q: node.op.name in {_g, _g + "dg"},
                respect_barriers=True,
            )
        # For IBM Eagle/Heron: x costs 2 sx pulses; fold into sx-count (weighted), sx-depth (unified).
        # For IBM Heron frac: report all physical 1Q gates together as 1q-count/1q-depth.
        if ibm_eagle or ibm_heron:
            xd, xc = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name == "x",
                respect_barriers=True,
            )
            p1_depth_combined, _ = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name in {"sx", "sxdg", "x"},
                respect_barriers=True,
            )
            p1_count_combined = p1_count + 2 * xc
            if p1_count_combined:
                data["sx-count"] = p1_count_combined
                data["sx-depth"] = p1_depth_combined
        elif ibm_heron_frac:
            oneq_depth, oneq_count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node: node.op.name in {"rx", "sx", "sxdg", "x"} and node.op.num_qubits == 1,
                respect_barriers=True,
            )
            if oneq_count:
                data["1q-count"] = oneq_count
                data["1q-depth"] = oneq_depth
        elif p1_count:
            data[f"{primitive_1q}-count"] = p1_count
            data[f"{primitive_1q}-depth"] = p1_depth
    else:
        # Generic: if there is exactly one distinct 1Q gate type that isn't a
        # known Clifford/rotation/measurement gate, report its count/depth.
        # This handles re-imported circuits with custom gates like phxz.
        _known_1q = _CLIFFORD_GATES | set(_ROTATION_GATES) | _NON_GATE_OPS | {"u", "u1", "u2", "u3"}
        unknown_1q = {
            instr.operation.name
            for instr in circuit.data
            if instr.operation.num_qubits == 1
            and instr.operation.name not in _known_1q
        }
        if len(unknown_1q) == 1:
            name_1q = next(iter(unknown_1q))
            p1_depth, p1_count = metric_depth_and_count(
                circuit,
                is_interesting=lambda node, _g=name_1q: node.op.name == _g,
                respect_barriers=True,
            )
            if p1_count:
                data[f"{name_1q}-count"] = p1_count
                data[f"{name_1q}-depth"] = p1_depth

    has_t = any(instr.operation.name in {"t", "tdg"} for instr in circuit.data)
    will_show_rot = not virtual_rz and not clifford_t and rc and has_parametric
    if (clifford_t or has_t) and not will_show_rot:
        tc, td = t_metrics(circuit)
        if tc:
            data["t-count"] = tc
            data["t-depth"] = td

    if will_show_rot:
        rot: dict = {
            "count": rc,
            "depth": rd,
            "breakdown": {
                (str(k) if k is not None else "approx"): v
                for k, v in rbreakdown.items()
            },
        }
        rot_t_count = sum(
            n * cnt for n, cnt in rbreakdown.items()
            if isinstance(n, int)
        )
        rot["t-count"] = rot_t_count
        if t_cost_fully_known(circuit):
            rot["t-depth"] = rt if rn_approx == 0 else None
        data["rotations"] = rot

    mcmc, mcmd = mcm_metrics(circuit)

    reset_dag = circuit_to_dag(circuit)
    resets = sum(
        1 for node in reset_dag.topological_op_nodes()
        if node.op.name == "reset"
        and any(
            isinstance(s, DAGOpNode) and s.op.name != "reset"
            for s in reset_dag.successors(node)
        )
    )

    if mcmc:
        data["mcm-count"] = mcmc
        data["mcm-depth"] = mcmd
    if resets:
        data["resets"] = resets

    return data


def print_costs(circuit, *, clifford_t: bool, cx_u: bool, cx_sx: bool, ecr_sx: bool, cz_sx: bool, iswap_rx: bool, rzz_rx: bool, rxx_rx: bool, xxphase_rx: bool, quantinuum_h: bool, ionq_aria: bool, ionq_forte: bool, syc_phxz: bool, sqrtiswap_phxz: bool, fez: bool, ibm_eagle: bool, ibm_heron: bool, ibm_heron_frac: bool, rigetti_ankaa: bool) -> None:
    data = collect_costs(circuit, clifford_t=clifford_t, cx_u=cx_u, cx_sx=cx_sx, ecr_sx=ecr_sx, cz_sx=cz_sx, iswap_rx=iswap_rx, rzz_rx=rzz_rx, rxx_rx=rxx_rx, xxphase_rx=xxphase_rx, quantinuum_h=quantinuum_h, ionq_aria=ionq_aria, ionq_forte=ionq_forte, syc_phxz=syc_phxz, sqrtiswap_phxz=sqrtiswap_phxz, fez=fez, ibm_eagle=ibm_eagle, ibm_heron=ibm_heron, ibm_heron_frac=ibm_heron_frac, rigetti_ankaa=rigetti_ankaa)

    rows: list[tuple[str, object] | None] = [
        ("width", data["width"]),
        ("depth", data["depth"]),
        ("gates", f"{data['gates']['display_count']}  ({data['gates']['breakdown']})"),
    ]
    if "clbits" in data:
        rows.append(("clbits", data["clbits"]))

    metric_rows: list[tuple[str, object] | None] = []

    if "2q-count" in data:
        metric_rows.append(("2q-count", data["2q-count"]))
        metric_rows.append(("2q-depth", data["2q-depth"]))
    # Named 2q gate type (e.g. cx-count, ecr-count, cz-count, ...)
    _known_non_gate_keys = {"2q-count", "sx-count", "t-count", "mcm-count"}
    for key in data:
        if key.endswith("-count") and key not in _known_non_gate_keys and key not in ("resets",):
            gate = key[:-len("-count")]
            metric_rows.append((f"{gate}-count", data[key]))
            metric_rows.append((f"{gate}-depth", data[f"{gate}-depth"]))
    if "sx-count" in data:
        metric_rows.append(("sx-count", data["sx-count"]))
        metric_rows.append(("sx-depth", data["sx-depth"]))
    if "t-count" in data:
        metric_rows.append(("t-count", data["t-count"]))
        metric_rows.append(("t-depth", data["t-depth"]))

    if "rotations" in data:
        rot = data["rotations"]
        clifford_count = rot["breakdown"].get("0", 0)
        non_clifford_count = rot["count"] - clifford_count
        count_str = f"{non_clifford_count}[+{clifford_count}]" if clifford_count else str(rot["count"])
        if "t-count" in rot:
            n_approx = rot["breakdown"].get("approx", 0)
            all_approx = n_approx == rot["count"] - clifford_count
            if all_approx:
                rot_count_str = count_str
            else:
                breakdown_str = format_rotation_breakdown(rot['breakdown'])
                rot_count_str = f"{count_str}  ({breakdown_str.lstrip('; ')})" if breakdown_str else count_str
        else:
            rot_count_str = count_str
        metric_rows.append(("rot-count", rot_count_str))
        if "t-depth" in rot and not all_approx:
            if rot["t-depth"] is None:
                t_depth_annotation = ""
            elif rot["t-depth"] > 0:
                t_depth_annotation = f"  (T-depth: {rot['t-depth']})"
            else:
                t_depth_annotation = ""
        else:
            t_depth_annotation = ""
        metric_rows.append(("rot-depth", f"{rot['depth']}{t_depth_annotation}"))

    measurement_rows: list[tuple[str, object] | None] = []
    if "mcm-count" in data:
        measurement_rows.append(("mcm-count", data["mcm-count"]))
        measurement_rows.append(("mcm-depth", data["mcm-depth"]))
    if "resets" in data:
        measurement_rows.append(("resets", data["resets"]))

    if metric_rows and measurement_rows:
        metric_rows.append(None)
    metric_rows.extend(measurement_rows)

    all_rows = rows + ([None] + metric_rows if metric_rows else [])
    header_width = max(len(row[0]) for row in rows if row is not None)
    metric_width = max((len(row[0]) for row in metric_rows if row is not None), default=0)
    for row in all_rows:
        if row is None:
            print()
        else:
            label, value = row
            width = metric_width if metric_rows and row in metric_rows else header_width
            print(f"{label.rjust(width)}: {value}")



def format_rotation_breakdown(breakdown: dict) -> str:
    """
    Format the rotation breakdown as a compact annotation string, e.g.
    ``clifford=2, T=3, T²=1, approx=1``.

    Accepts either int|None keys (from rotation_metrics) or str keys
    (from collect_costs JSON-safe form).

    Dyadic orders > 1 are written using Unicode superscripts.
    """
    _super = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    def superscript(n: int) -> str:
        return str(n).translate(_super)

    # Normalise keys to int|None.
    def normalise_key(k):
        if k == "approx" or k is None:
            return None
        return int(k)

    norm = {normalise_key(k): v for k, v in breakdown.items()}
    non_clifford = []
    for n in sorted(k for k in norm if isinstance(k, int) and k >= 1):
        label = "T" if n == 1 else f"T{superscript(n)}"
        non_clifford.append(f"{label}={norm[n]}")
    if norm.get(None, 0):
        non_clifford.append(f"approx={norm[None]}")

    clifford = []
    if norm.get(0, 0):
        clifford.append(f"clifford={norm[0]}")

    sections = [", ".join(non_clifford)] if non_clifford else []
    if clifford:
        sections.append(", ".join(clifford))
    return "; " + "; ".join(sections) if sections else ""

def _cirq_op_to_qasm_line(op, qubit_to_name: dict, precision: int = 10) -> str:
    """
    Emit a single QASM 2 instruction for a Cirq operation.
    ``qubit_to_name`` maps each Cirq qubit to its QASM string name,
    e.g. ``"q[0]"`` for circuit body or ``"q0"`` for gate definitions.
    """
    import cirq
    import math as _math

    gate = op.gate
    qs = [qubit_to_name[q] for q in op.qubits]

    def fmt(v):
        # Format a float as a compact QASM number
        return f"{v:.{precision}g}"

    if isinstance(gate, cirq.PhasedXZGate):
        x = gate.x_exponent
        z = gate.z_exponent
        a = gate.axis_phase_exponent
        # PhasedXZGate = Z^{-a} X^x Z^{a} Z^z = rz(-a*pi) rx(x*pi) rz((a+z)*pi)
        # In u3 form: u3(theta, phi, lam) = rz(phi+pi/2) ry(theta) rz(lam-pi/2)  -- not quite
        # Emit as a named phxz gate call with parameters
        a_val = fmt(float(a))
        x_val = fmt(float(x))
        z_val = fmt(float(z))
        return f"phxz({a_val},{x_val},{z_val}) {qs[0]};"

    if isinstance(gate, cirq.MeasurementGate):
        key = gate.key
        return "\n".join(f"measure {qs[i]} -> m_{key}[{i}];" for i in range(len(qs)))

    # Named gate fallback: use gate's own _qasm_ if available, else error
    name_map = {
        cirq.XPowGate: lambda g: (f"rx({fmt(float(g.exponent)*_math.pi)})", 1),
        cirq.YPowGate: lambda g: (f"ry({fmt(float(g.exponent)*_math.pi)})", 1),
        cirq.ZPowGate: lambda g: (f"rz({fmt(float(g.exponent)*_math.pi)})", 1),
        cirq.CZPowGate: lambda g: ("cz", 2) if float(g.exponent) == 1.0 else None,
    }
    for cls, fn in name_map.items():
        if isinstance(gate, cls):
            result = fn(gate)
            if result is not None:
                instr, _ = result
                return f"{instr} {', '.join(qs)};"

    # Try cirq's own _qasm_ protocol
    args = cirq.QasmArgs(
        precision=precision,
        qubit_id_map={q: qs[i] for i, q in enumerate(op.qubits)},
        version="2.0",
    )
    qasm_str = cirq.qasm(op, args=args, default=None)
    if qasm_str is not None:
        return qasm_str.strip()

    raise ValueError(f"Don't know how to emit QASM for {op}")


def _make_gate_definition(gate_name: str, body_ops_qasm: list[str], n_qubits: int) -> str:
    """Emit an OpenQASM 2 gate definition block."""
    qubit_names = [f"q{i}" for i in range(n_qubits)]
    qubit_list = ", ".join(qubit_names)
    body = "\n  ".join(body_ops_qasm)
    return f"gate {gate_name} {qubit_list} {{\n  {body}\n}}"


PHXZ_GATE_DEF = (
    "gate phxz(a,x,z) q0 {\n"
    "  rz(-a*pi) q0;\n"
    "  rx(x*pi) q0;\n"
    "  rz((a+z)*pi) q0;\n"
    "}"
)


def _compile_cirq(qc, *, gateset_name: str):
    """
    Compile a Qiskit QuantumCircuit to a Google native gateset using Cirq,
    returning (compiled_cirq_circuit, qiskit_circuit).

    ``gateset_name`` is either ``"syc"`` or ``"sqrtiswap"``.

    The compiled Cirq circuit is returned for cost analysis (counting
    PhasedXZGate / SycamoreGate / SQRT_ISWAP).  The Qiskit circuit is
    re-parsed from hand-crafted QASM that keeps the native gate names
    (syc / sqrt_iswap / phxz) with gate definitions at the top, exactly
    like Qiskit does for the ecr gate.

    Raises ``SystemExit`` with a friendly message if cirq is not installed.
    """
    try:
        import cirq
        import cirq_google
    except ImportError:
        raise SystemExit("--syc-phxz/--sqrtiswap-phxz require cirq: pip install cirq")

    import math as _math
    import numpy as _np
    from qiskit import qasm2 as qiskit_qasm2

    from cirq.contrib.qasm_import import circuit_from_qasm as cirq_circuit_from_qasm

    # Qiskit → QASM 2 → Cirq.
    # Cirq's contrib QASM parser only knows qelib1.inc gates. Replace any
    # 'p(...)' (PhaseGate, qelib1-absent but equivalent to rz up to global
    # phase) with 'rz(...)' so the parser accepts it.
    qasm2_str = qiskit_qasm2.dumps(qc)
    qasm2_str = re.sub(r'\bp\(', 'rz(', qasm2_str)
    cirq_circuit = cirq_circuit_from_qasm(qasm2_str)

    # Compile to target gateset. Suppress a Cirq-internal FutureWarning
    # about use_repetition_ids that is triggered inside optimize_for_target_gateset.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=r".*use_repetition_ids.*",
            category=FutureWarning,
            module=r"cirq.*",
        )
        if gateset_name == "syc":
            gateset = cirq_google.SycamoreTargetGateset()
            twoq_gate_name = "syc"
        else:
            gateset = cirq.SqrtIswapTargetGateset()
            twoq_gate_name = "sqrt_iswap"
        compiled_cirq = cirq.optimize_for_target_gateset(cirq_circuit, gateset=gateset)

    # --- Step 3: build QASM manually, keeping native gate names ---
    # Sort qubits for stable ordering
    all_qubits = sorted(compiled_cirq.all_qubits())
    n = len(all_qubits)
    qubit_to_name = {q: f"q[{i}]" for i, q in enumerate(all_qubits)}

    # Compute gate definition bodies using Cirq's own KAK decomposition
    # by decomposing onto two fresh LineQubits and emitting the sub-ops.
    def _gate_def_body(gate, n_qubits):
        """Decompose gate to CZ+PhasedXZ and emit as rz/rx/cz QASM lines."""
        lq = cirq.LineQubit.range(n_qubits)
        dummy_op = gate(*lq)
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=r".*use_repetition_ids.*", category=FutureWarning)
            decomposed = cirq.Circuit(
                cirq.optimize_for_target_gateset(
                    cirq.Circuit(dummy_op),
                    gateset=cirq.CZTargetGateset(),
                )
            )
        body_lines = []
        # Gate definition bodies use bare parameter names q0, q1, ...
        def_qubit_to_name = {q: f"q{i}" for i, q in enumerate(lq)}
        for moment in decomposed:
            for op in moment.operations:
                body_lines.append(_cirq_op_to_qasm_line(op, def_qubit_to_name))
        return body_lines

    # Collect which special gates actually appear
    needs_syc = any(
        isinstance(op.gate, cirq_google.SycamoreGate)
        for moment in compiled_cirq for op in moment.operations
    ) if gateset_name == "syc" else False
    needs_sqrtiswap = any(
        isinstance(op.gate, (cirq.SQRT_ISWAP_INV.__class__, cirq.ISwapPowGate))
        and abs(abs(float(op.gate.exponent)) - 0.5) < 1e-6
        for moment in compiled_cirq for op in moment.operations
    ) if gateset_name == "sqrtiswap" else False

    # Build gate definitions — phxz is always needed for PhasedXZGate
    gate_defs = [PHXZ_GATE_DEF]
    if needs_syc:
        body = _gate_def_body(cirq_google.SYC, 2)
        gate_defs.append(_make_gate_definition("syc", body, 2))
    if needs_sqrtiswap:
        body = _gate_def_body(cirq.SQRT_ISWAP, 2)
        gate_defs.append(_make_gate_definition("sqrt_iswap", body, 2))

    # Collect measurement keys
    meas_keys = {}
    for moment in compiled_cirq:
        for op in moment.operations:
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                if key not in meas_keys:
                    meas_keys[key] = len(op.qubits)

    # Emit QASM
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
    ]
    if gate_defs:
        lines.append("")
        lines.extend(gate_defs)
    lines.append("")
    lines.append(f"qreg q[{n}];")
    for key, nbits in meas_keys.items():
        lines.append(f"creg m_{key}[{nbits}];")
    lines.append("")

    for moment in compiled_cirq:
        for op in moment.operations:
            gate = op.gate
            qs = [qubit_to_name[q] for q in op.qubits]

            if isinstance(gate, cirq.MeasurementGate):
                for i, q in enumerate(op.qubits):
                    lines.append(f"measure {qubit_to_name[q]} -> m_{gate.key}[{i}];")
            elif gateset_name == "syc" and isinstance(gate, cirq_google.SycamoreGate):
                lines.append(f"syc {qs[0]}, {qs[1]};")
            elif gateset_name == "sqrtiswap" and isinstance(gate, cirq.ISwapPowGate) and abs(float(gate.exponent) - 0.5) < 1e-6:
                lines.append(f"sqrt_iswap {qs[0]}, {qs[1]};")
            elif gateset_name == "sqrtiswap" and isinstance(gate, cirq.ISwapPowGate) and abs(float(gate.exponent) + 0.5) < 1e-6:
                lines.append(f"sqrt_iswap_inv {qs[0]}, {qs[1]};")
            else:
                lines.append(_cirq_op_to_qasm_line(op, qubit_to_name))

    qasm2_out = "\n".join(lines)

    # --- Step 4: parse back to Qiskit ---
    # Teach the parser about 'phxz(a,x,z)' which is our custom 1Q gate.
    from qiskit.qasm2 import CustomInstruction
    from qiskit.circuit.library import RZGate, RXGate
    from qiskit.circuit import QuantumCircuit as _QC, Gate as _Gate
    import numpy as _np2

    class PhXZGate(_Gate):
        def __init__(self, a, x, z):
            super().__init__("phxz", 1, [a, x, z])
        def _define(self):
            qc = _QC(1)
            a, x, z = self.params
            qc.rz(-a * _np2.pi, 0)
            qc.rx(x * _np2.pi, 0)
            qc.rz((a + z) * _np2.pi, 0)
            self.definition = qc
        def to_matrix(self):
            a, x, z = (float(p) for p in self.params)
            # PhasedXZGate = rz(-a*pi) rx(x*pi) rz((a+z)*pi)  (left = last applied)
            cx, sx = _np2.cos(x * _np2.pi / 2), _np2.sin(x * _np2.pi / 2)
            rz1 = _np2.array([[_np2.exp(1j * a * _np2.pi / 2), 0],
                               [0, _np2.exp(-1j * a * _np2.pi / 2)]])
            rx  = _np2.array([[cx, -1j * sx], [-1j * sx, cx]])
            az = (a + z) * _np2.pi
            rz2 = _np2.array([[_np2.exp(-1j * az / 2), 0],
                               [0, _np2.exp(1j * az / 2)]])
            return rz2 @ rx @ rz1

    phxz_instruction = CustomInstruction("phxz", 3, 1, PhXZGate)
    qiskit_circuit = qiskit_qasm2.loads(qasm2_out, custom_instructions=[phxz_instruction])

    return compiled_cirq, qiskit_circuit, qasm2_out


# ---------------------------------------------------------------------------
# pytket-based compilation
# ---------------------------------------------------------------------------

# Gate definitions for non-qelib1 gates that pytket may produce.
# Parameters are in half-turns (multiples of pi) as pytket uses.
_PYTKET_GATE_DEFS = {
    # PhasedX(alpha, beta) = Rz(beta*pi) Rx(alpha*pi) Rz(-beta*pi)
    "phasedx": (
        "gate phasedx(alpha, beta) q0 {\n"
        "  rz(-beta*pi) q0;\n"
        "  rx(alpha*pi) q0;\n"
        "  rz(beta*pi) q0;\n"
        "}"
    ),
    # ZZPhase(alpha) = exp(-i * alpha*pi/2 * ZZ)
    "zzphase": (
        "gate zzphase(alpha) q0, q1 {\n"
        "  cx q0, q1;\n"
        "  rz(alpha*pi) q1;\n"
        "  cx q0, q1;\n"
        "}"
    ),
    # ZZMax = ZZPhase(0.5) = exp(-i*pi/4 * ZZ)
    "zzmax": (
        "gate zzmax q0, q1 {\n"
        "  cx q0, q1;\n"
        "  rz(pi/2) q1;\n"
        "  cx q0, q1;\n"
        "}"
    ),
    # XXPhase(alpha) = exp(-i * alpha*pi/2 * XX)
    "xxphase": (
        "gate xxphase(alpha) q0, q1 {\n"
        "  h q0;\n"
        "  h q1;\n"
        "  cx q0, q1;\n"
        "  rz(alpha*pi) q1;\n"
        "  cx q0, q1;\n"
        "  h q0;\n"
        "  h q1;\n"
        "}"
    ),
}


def _pytket_op_to_qasm_line(cmd, qubit_to_name: dict, precision: int = 10) -> str | None:
    """
    Emit a single QASM 2 instruction for a pytket Command.
    Returns None for barrier/phase/noop operations that should be skipped.
    All pytket angles are in half-turns; we multiply by pi for QASM radian args.
    """
    import math as _math
    from pytket import OpType

    op = cmd.op
    optype = op.type
    qs = [qubit_to_name[q] for q in cmd.qubits]

    def fmt(v):
        # Format a float as a compact QASM number (already in radians)
        return f"{v:.{precision}g}"

    def ht_to_rad(v):
        # Half-turn to radian
        return float(v) * _math.pi

    params = op.params  # half-turns

    # Standard qelib1 single-qubit gates
    if optype == OpType.Rz:
        return f"rz({fmt(ht_to_rad(params[0]))}) {qs[0]};"
    if optype == OpType.Rx:
        return f"rx({fmt(ht_to_rad(params[0]))}) {qs[0]};"
    if optype == OpType.Ry:
        return f"ry({fmt(ht_to_rad(params[0]))}) {qs[0]};"
    if optype == OpType.H:
        return f"h {qs[0]};"
    if optype == OpType.X:
        return f"x {qs[0]};"
    if optype == OpType.Y:
        return f"y {qs[0]};"
    if optype == OpType.Z:
        return f"z {qs[0]};"
    if optype == OpType.S:
        return f"s {qs[0]};"
    if optype == OpType.Sdg:
        return f"sdg {qs[0]};"
    if optype == OpType.T:
        return f"t {qs[0]};"
    if optype == OpType.Tdg:
        return f"tdg {qs[0]};"
    if optype == OpType.SX:
        return f"sx {qs[0]};"
    if optype == OpType.SXdg:
        return f"sxdg {qs[0]};"
    if optype == OpType.U1:
        return f"u1({fmt(ht_to_rad(params[0]))}) {qs[0]};"
    if optype == OpType.U2:
        return f"u2({fmt(ht_to_rad(params[0]))},{fmt(ht_to_rad(params[1]))}) {qs[0]};"
    if optype == OpType.U3:
        return f"u3({fmt(ht_to_rad(params[0]))},{fmt(ht_to_rad(params[1]))},{fmt(ht_to_rad(params[2]))}) {qs[0]};"

    # Non-qelib1 single-qubit gates
    if optype == OpType.PhasedX:
        # phasedx(alpha, beta) — body emits rz/rx/rz
        return f"phasedx({fmt(params[0])},{fmt(params[1])}) {qs[0]};"

    # Standard 2Q gates
    if optype == OpType.CX:
        return f"cx {qs[0]}, {qs[1]};"
    if optype == OpType.CZ:
        return f"cz {qs[0]}, {qs[1]};"
    if optype == OpType.ECR:
        return f"ecr {qs[0]}, {qs[1]};"
    if optype == OpType.ISWAPMax:
        return f"iswap {qs[0]}, {qs[1]};"
    if optype == OpType.ZZMax:
        return f"zzmax {qs[0]}, {qs[1]};"

    # Non-qelib1 2Q gates — use our custom gate names
    if optype == OpType.ZZPhase:
        return f"zzphase({fmt(params[0])}) {qs[0]}, {qs[1]};"
    if optype == OpType.XXPhase:
        return f"xxphase({fmt(params[0])}) {qs[0]}, {qs[1]};"

    # Measurements
    if optype == OpType.Measure:
        bit = cmd.bits[0]
        bit_name = f"m_{bit.reg_name}[{bit.index[0]}]"
        return f"measure {qs[0]} -> {bit_name};"

    # Barriers, noop, global phase — skip
    if optype in (OpType.Barrier, OpType.noop, OpType.Phase):
        return None

    raise ValueError(f"_compile_pytket: don't know how to emit QASM for {cmd}")


def _compile_pytket(qc, *, rebase_pass, gateset_name: str):
    """
    Compile a Qiskit QuantumCircuit using pytket, returning
    (pytket_circuit, qiskit_circuit, qasm2_out).

    ``rebase_pass`` is a callable pytket pass (e.g. AutoRebase instance)
    that accepts a pytket Circuit and rebases it to the target gate set.

    ``gateset_name`` is a short string used in error messages.

    The QASM2 output uses hand-crafted gate definitions for any non-qelib1
    gates in the compiled circuit (PhasedX, ZZPhase, XXPhase, etc.).

    Raises SystemExit if pytket is not installed.
    """
    try:
        from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
        from pytket import OpType
    except ImportError:
        raise SystemExit(
            f"--{gateset_name} requires pytket: pip install pytket"
        )

    from qiskit import qasm2 as qiskit_qasm2

    # --- Step 1: Qiskit → QASM2 → pytket ---
    qasm2_str = qiskit_qasm2.dumps(qc)
    qasm2_str = re.sub(r'\bp\(', 'rz(', qasm2_str)  # p( not in qelib1
    tk_circuit = circuit_from_qasm_str(qasm2_str)

    # --- Step 2: rebase ---
    rebase_pass.apply(tk_circuit)

    # --- Step 3: emit QASM2 with gate definitions ---
    # Determine which non-qelib1 gates appear
    needed_defs = set()
    _PHASEDX_TYPES = {OpType.PhasedX}
    _ZZPHASE_TYPES = {OpType.ZZPhase}
    _ZZMAX_TYPES = {OpType.ZZMax}
    _XXPHASE_TYPES = {OpType.XXPhase}
    for cmd in tk_circuit.get_commands():
        if cmd.op.type in _PHASEDX_TYPES:
            needed_defs.add("phasedx")
        elif cmd.op.type in _ZZPHASE_TYPES:
            needed_defs.add("zzphase")
        elif cmd.op.type in _ZZMAX_TYPES:
            needed_defs.add("zzmax")
        elif cmd.op.type in _XXPHASE_TYPES:
            needed_defs.add("xxphase")

    # Build qubit name map
    all_qubits = tk_circuit.qubits
    qubit_to_name = {q: f"q[{i}]" for i, q in enumerate(all_qubits)}

    # Collect classical registers for measurements
    creg_sizes: dict[str, int] = {}
    for cmd in tk_circuit.get_commands():
        from pytket import OpType as _OT
        if cmd.op.type == _OT.Measure:
            bit = cmd.bits[0]
            reg = bit.reg_name
            idx = bit.index[0]
            creg_sizes[reg] = max(creg_sizes.get(reg, 0), idx + 1)

    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";']
    if needed_defs:
        lines.append("")
        for gate_name in sorted(needed_defs):
            lines.append(_PYTKET_GATE_DEFS[gate_name])
    lines.append("")
    lines.append(f"qreg q[{len(all_qubits)}];")
    for reg, size in creg_sizes.items():
        lines.append(f"creg m_{reg}[{size}];")
    lines.append("")

    for cmd in tk_circuit.get_commands():
        line = _pytket_op_to_qasm_line(cmd, qubit_to_name)
        if line is not None:
            lines.append(line)

    qasm2_out = "\n".join(lines)

    # --- Step 4: QASM2 → Qiskit ---
    qiskit_circuit = qiskit_qasm2.loads(qasm2_out)

    return tk_circuit, qiskit_circuit, qasm2_out


# QASM 2 gate definitions for IonQ native gates.
# GPI(phi)  = a phased π pulse:   rz(-phi*2π) rx(π) rz(phi*2π)
# GPI2(phi) = a phased π/2 pulse: rz(-phi*2π) rx(π/2) rz(phi*2π)
# MS(phi0,phi1) = Mølmer–Sørensen: a fixed-angle entangling gate with per-qubit phases
# ZZ(theta) = ZZ rotation: a fixed-angle ZZ entangling gate (Forte native)
_GPI_GATE_DEF = (
    "gate gpi(phi) q0 {\n"
    "  rz(-phi*2*pi) q0;\n"
    "  rx(pi) q0;\n"
    "  rz(phi*2*pi) q0;\n"
    "}"
)
_GPI2_GATE_DEF = (
    "gate gpi2(phi) q0 {\n"
    "  rz(-phi*2*pi) q0;\n"
    "  rx(pi/2) q0;\n"
    "  rz(phi*2*pi) q0;\n"
    "}"
)
# MS(phi0,phi1) decomposes entirely into qelib1 gates:
#   MS(phi0,phi1) = rz(phi0*2π)⊗rz(phi1*2π) · H⊗H · CX · rz(π/2) · CX · H⊗H · rz(-phi0*2π)⊗rz(-phi1*2π)
_MS_GATE_DEF = (
    "gate ms(phi0,phi1) q0,q1 {\n"
    "  rz(-phi0*2*pi) q0;\n"
    "  rz(-phi1*2*pi) q1;\n"
    "  h q0;\n"
    "  h q1;\n"
    "  cx q0,q1;\n"
    "  rz(pi/2) q1;\n"
    "  cx q0,q1;\n"
    "  h q0;\n"
    "  h q1;\n"
    "  rz(phi0*2*pi) q0;\n"
    "  rz(phi1*2*pi) q1;\n"
    "}"
)
# ZZ(theta) = exp(-i*theta*π * Z⊗Z), theta in turns.
# Decomposes into qelib1 as: cx q0,q1; rz(theta*2π) q1; cx q0,q1
_ZZ_GATE_DEF = (
    "gate zz(theta) q0,q1 {\n"
    "  cx q0,q1;\n"
    "  rz(theta*2*pi) q1;\n"
    "  cx q0,q1;\n"
    "}"
)


def _compile_ionq(qc, *, gateset_name: str):
    """
    Compile a Qiskit QuantumCircuit to an IonQ hardware-native gateset using
    cirq-ionq, returning (compiled_cirq_circuit, qiskit_circuit, qasm2_out).

    ``gateset_name`` is either ``"aria"`` (GPI, GPI2, MS) or ``"forte"`` (GPI, GPI2, ZZ).

    The compiled Cirq circuit is returned for cost analysis.  The Qiskit circuit
    is re-parsed from hand-crafted QASM 2 that keeps the native gate names with
    definitions at the top, exactly as _compile_cirq does for syc/sqrt_iswap.

    Raises ``SystemExit`` with a friendly message if cirq or cirq-ionq are not
    installed, or if cirq-ionq does not yet ship AriaNativeGateset/ForteNativeGateset.
    """
    try:
        import cirq
    except ImportError:
        raise SystemExit("--ionq-aria/--ionq-forte require cirq: pip install cirq")
    try:
        import cirq_ionq
        from cirq_ionq.ionq_native_target_gateset import AriaNativeGateset, ForteNativeGateset
        from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate, ZZGate
    except ImportError:
        raise SystemExit("--ionq-aria/--ionq-forte require cirq-ionq: pip install cirq-ionq")
    except AttributeError:
        raise SystemExit(
            "--ionq-aria/--ionq-forte require cirq-ionq >= 1.5 with AriaNativeGateset. "
            "Install a recent version: pip install --upgrade cirq-ionq"
        )

    from qiskit import qasm2 as qiskit_qasm2
    from cirq.contrib.qasm_import import circuit_from_qasm as cirq_circuit_from_qasm

    # Qiskit → QASM 2 → Cirq.
    qasm2_str = qiskit_qasm2.dumps(qc)
    qasm2_str = re.sub(r'\bp\(', 'rz(', qasm2_str)
    cirq_circuit = cirq_circuit_from_qasm(qasm2_str)

    # Compile to IonQ native gateset.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=r".*use_repetition_ids.*",
            category=FutureWarning,
            module=r"cirq.*",
        )
        if gateset_name == "aria":
            gateset = AriaNativeGateset()
        else:
            gateset = ForteNativeGateset()
        compiled_cirq = cirq.optimize_for_target_gateset(cirq_circuit, gateset=gateset)

    # Build hand-crafted QASM 2, keeping native gate names.
    all_qubits = sorted(compiled_cirq.all_qubits())
    n = len(all_qubits)
    qubit_to_name = {q: f"q[{i}]" for i, q in enumerate(all_qubits)}

    # Collect measurement keys.
    meas_keys: dict[str, int] = {}
    for moment in compiled_cirq:
        for op in moment.operations:
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                if key not in meas_keys:
                    meas_keys[key] = len(op.qubits)

    def _fmt(v: float, precision: int = 10) -> str:
        return f"{v:.{precision}g}"

    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        '',
        _GPI_GATE_DEF,
        _GPI2_GATE_DEF,
    ]
    if gateset_name == "aria":
        lines.append(_MS_GATE_DEF)
    else:
        lines.append(_ZZ_GATE_DEF)
    lines.append('')
    lines.append(f"qreg q[{n}];")
    for key, nbits in meas_keys.items():
        lines.append(f"creg m_{key}[{nbits}];")
    lines.append('')

    for moment in compiled_cirq:
        for op in moment.operations:
            gate = op.gate
            qs = [qubit_to_name[q] for q in op.qubits]

            if isinstance(gate, cirq.MeasurementGate):
                for i, q in enumerate(op.qubits):
                    lines.append(f"measure {qubit_to_name[q]} -> m_{gate.key}[{i}];")
            elif isinstance(gate, GPIGate):
                lines.append(f"gpi({_fmt(float(gate.phi))}) {qs[0]};")
            elif isinstance(gate, GPI2Gate):
                lines.append(f"gpi2({_fmt(float(gate.phi))}) {qs[0]};")
            elif isinstance(gate, MSGate):
                lines.append(
                    f"ms({_fmt(float(gate.phi0))},{_fmt(float(gate.phi1))}) {qs[0]},{qs[1]};"
                )
            elif isinstance(gate, ZZGate):
                lines.append(f"zz({_fmt(float(gate.theta))}) {qs[0]},{qs[1]};")
            else:
                lines.append(_cirq_op_to_qasm_line(op, qubit_to_name))

    qasm2_out = "\n".join(lines)

    # Parse back to Qiskit for visualisation / simulation.
    from qiskit.qasm2 import CustomInstruction
    from qiskit.circuit import QuantumCircuit as _QC, Gate as _Gate
    import numpy as _np2

    class GPIQiskitGate(_Gate):
        def __init__(self, phi):
            super().__init__("gpi", 1, [phi])
        def _define(self):
            qc = _QC(1)
            phi = float(self.params[0])
            qc.rz(-phi * 2 * _np2.pi, 0)
            qc.rx(_np2.pi, 0)
            qc.rz(phi * 2 * _np2.pi, 0)
            self.definition = qc

    class GPI2QiskitGate(_Gate):
        def __init__(self, phi):
            super().__init__("gpi2", 1, [phi])
        def _define(self):
            qc = _QC(1)
            phi = float(self.params[0])
            qc.rz(-phi * 2 * _np2.pi, 0)
            qc.rx(_np2.pi / 2, 0)
            qc.rz(phi * 2 * _np2.pi, 0)
            self.definition = qc

    class MSQiskitGate(_Gate):
        def __init__(self, phi0, phi1):
            super().__init__("ms", 2, [phi0, phi1])
        def _define(self):
            qc = _QC(2)
            phi0, phi1 = float(self.params[0]), float(self.params[1])
            qc.rz(-phi0 * 2 * _np2.pi, 0)
            qc.rz(-phi1 * 2 * _np2.pi, 1)
            qc.h(0)
            qc.h(1)
            qc.cx(0, 1)
            qc.rz(_np2.pi / 2, 1)
            qc.cx(0, 1)
            qc.h(0)
            qc.h(1)
            qc.rz(phi0 * 2 * _np2.pi, 0)
            qc.rz(phi1 * 2 * _np2.pi, 1)
            self.definition = qc

    class ZZQiskitGate(_Gate):
        def __init__(self, theta):
            super().__init__("zz", 2, [theta])
        def _define(self):
            qc = _QC(2)
            theta = float(self.params[0])
            qc.cx(0, 1)
            qc.rz(theta * 2 * _np2.pi, 1)
            qc.cx(0, 1)
            self.definition = qc

    custom_instructions = [
        CustomInstruction("gpi", 1, 1, GPIQiskitGate),
        CustomInstruction("gpi2", 1, 1, GPI2QiskitGate),
    ]
    if gateset_name == "aria":
        custom_instructions.append(CustomInstruction("ms", 2, 2, MSQiskitGate))
    else:
        custom_instructions.append(CustomInstruction("zz", 1, 2, ZZQiskitGate))

    qiskit_circuit = qiskit_qasm2.loads(qasm2_out, custom_instructions=custom_instructions)

    return compiled_cirq, qiskit_circuit, qasm2_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read OpenQASM 3 from files or stdin, optionally transpile each "
            "input circuit, then visualize, print costs, dump, and/or simulate "
            "the selected circuit. The selected circuit is the original input "
            "by default. If a target flag is given, the selected circuit is the "
            "transpiled result instead."
        )
    )
    parser.add_argument(
        "qasm_files",
        nargs="*",
        metavar="qasm_file",
        help="`.qasm` files to read. Each input is processed independently. Reads from `stdin` if omitted.",
    )

    parser.add_argument_group("target selection (choose at most one)")

    target_portable = parser.add_argument_group("portable / logical targets")
    target_portable.add_argument(
        "--cx-u",
        action="store_true",
        dest="cx_u",
        help="transpile into the basis {cx, u}.",
    )
    target_portable.add_argument(
        "--clifford-t",
        action="store_true",
        dest="clifford_t",
        help="transpile into the Clifford+T basis {cx, h, s, sdg, t, tdg}.",
    )

    target_sc = parser.add_argument_group(
        "hardware-family proxy targets",
        "These are reduced compilation / cost-model targets. They preserve a hardware family's "
        "dominant entangler and 1q rotation style, but they are not always the vendor's full "
        "documented native gate set.",
    )
    target_sc.add_argument(
        "--cx-sx",
        action="store_true",
        dest="cx_sx",
        help="transpile into the proxy basis {rz, sx, cx}. Generic CX-based superconducting proxy.",
    )
    target_sc.add_argument(
        "--cz-sx",
        action="store_true",
        dest="cz_sx",
        help="transpile into the proxy basis {rz, sx, cz}. CZ-family superconducting proxy.",
    )
    target_sc.add_argument(
        "--ecr-sx",
        action="store_true",
        dest="ecr_sx",
        help="transpile into the proxy basis {rz, sx, ecr}. Cross-resonance / ECR-family proxy.",
    )
    target_sc.add_argument(
        "--iswap-rx",
        action="store_true",
        dest="iswap_rx",
        help="transpile into the proxy basis {rz, rx, iswap}. iSWAP-family proxy.",
    )
    target_sc.add_argument(
        "--rzz-rx",
        action="store_true",
        dest="rzz_rx",
        help="transpile into the proxy basis {rz, rx, rzz}. Parameterized-ZZ proxy.",
    )
    target_sc.add_argument(
        "--rxx-rx",
        action="store_true",
        dest="rxx_rx",
        help="transpile into the proxy basis {rz, rx, rxx}. Parameterized-XX proxy.",
    )
    target_sc.add_argument(
        "--sqrtiswap-phxz",
        action="store_true",
        dest="sqrtiswap_phxz",
        help="transpile into the proxy basis {sqrt_iswap, phxz}. sqrt-iSWAP / XY-family proxy. Requires `cirq`.",
    )
    target_sc.add_argument(
        "--xxphase-rx",
        action="store_true",
        dest="xxphase_rx",
        help="transpile into the proxy basis {rz, rx, xxphase}. Requires `pytket`.",
    )

    target_vendor = parser.add_argument_group("exact vendor targets")
    target_vendor.add_argument(
        "--ibm-eagle",
        action="store_true",
        dest="ibm_eagle",
        help="transpile into the IBM Eagle (ECR-family) basis {ecr, id, rz, sx, x}.",
    )
    target_vendor.add_argument(
        "--ibm-heron",
        action="store_true",
        dest="ibm_heron",
        help="transpile into the IBM Heron basis {cz, id, rz, sx, x}.",
    )
    target_vendor.add_argument(
        "--ibm-heron-frac",
        action="store_true",
        dest="ibm_heron_frac",
        help="transpile into the IBM Heron fractional-gate basis {cz, id, rz, sx, x, rx, rzz}.",
    )
    target_vendor.add_argument(
        "--rigetti-ankaa",
        action="store_true",
        dest="rigetti_ankaa",
        help="transpile into the Rigetti Ankaa basis {rx, rz, iswap}.",
    )
    target_vendor.add_argument(
        "--quantinuum-h",
        action="store_true",
        dest="quantinuum_h",
        help="transpile into the Quantinuum H-series basis {rz, phasedx, zzphase}. Requires `pytket`.",
    )
    target_vendor.add_argument(
        "--google-sycamore",
        action="store_true",
        dest="google_sycamore",
        help="transpile into the Google Sycamore target {syc, phxz}. Requires `cirq`.",
    )
    target_vendor.add_argument(
        "--google-sqrtiswap",
        action="store_true",
        dest="google_sqrtiswap",
        help="transpile into the Google sqrt-iSWAP target {sqrt_iswap, phxz}. Requires `cirq`.",
    )
    target_vendor.add_argument(
        "--ionq-aria",
        action="store_true",
        dest="ionq_aria",
        help="transpile into the IonQ Aria native basis {gpi, gpi2, ms}. Requires `cirq` and `cirq-ionq`.",
    )
    target_vendor.add_argument(
        "--ionq-forte",
        action="store_true",
        dest="ionq_forte",
        help="transpile into the IonQ Forte native basis {gpi, gpi2, zz}. Requires `cirq` and `cirq-ionq`.",
    )

    target_exec = parser.add_argument_group("execution-ready backend targets")
    target_exec.add_argument(
        "--fez",
        action="store_true",
        help="transpile for FakeFez and use the execution-ready circuit.",
    )

    transpile_group = parser.add_argument_group("transpilation")
    transpile_group.add_argument(
        "--opt",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        dest="opt_level",
        help="transpiler optimization level (default: 3).",
    )
    transpile_group.add_argument(
        "--eps",
        type=float,
        help="maximum allowed approximation error for Clifford+T synthesis. Requires `--clifford-t`.",
    )
    transpile_group.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="N",
        help="maximum qubit width available, including ancillas. The optimizer may use extra qubits up to this limit to reduce gate count or depth.",
    )

    output_group = parser.add_argument_group("output control")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="emit cost metrics as JSON. With `--run`, emit simulation results as JSON instead.",
    )
    output_group.add_argument(
        "--show",
        action="store_true",
        help="visualize the selected circuit.",
    )
    output_group.add_argument(
        "--dump",
        action="store_true",
        help="dump the selected circuit as OpenQASM 3.",
    )
    output_group.add_argument(
        "--dump2",
        action="store_true",
        help="dump the selected circuit as OpenQASM 2.",
    )
    output_group.add_argument(
        "--run",
        action="store_true",
        help="simulate the selected circuit with qblaze and print the classical bits and final sparse state.",
    )
    output_group.add_argument(
        "--cost",
        action="store_true",
        help="print operation counts even when not visualizing the circuit.",
    )
    output_group.add_argument(
        "--no-cost",
        action="store_true",
        help="do not print operation counts, including after `--show`.",
    )
    output_group.add_argument(
        "--fold",
        nargs="?",
        type=int,
        const=None,
        default=-1,
        help=(
            "circuit draw fold width. Default: `-1` (no folding). "
            "Use `--fold` for automatic width detection."
        ),
    )
    args = parser.parse_args()

    _targets = [name for name in (
        "cx_u", "clifford_t", "cx_sx", "ecr_sx", "cz_sx",
        "iswap_rx", "rzz_rx", "rxx_rx", "xxphase_rx", "sqrtiswap_phxz", "fez",
        "ibm_eagle", "ibm_heron", "ibm_heron_frac", "rigetti_ankaa",
        "quantinuum_h", "google_sycamore", "google_sqrtiswap",
        "ionq_aria", "ionq_forte",
    ) if getattr(args, name)]
    if len(_targets) > 1:
        parser.error(f"at most one target may be specified, got: {', '.join('--' + t.replace('_', '-') for t in _targets)}")

    if args.eps is not None and not args.clifford_t:
        parser.error("--eps requires --clifford-t")
    if args.eps is not None and args.eps < 0:
        parser.error("--eps must be non-negative")
    if args.eps == 0:
        args.eps = None
    if args.width is not None and args.width < 1:
        parser.error("--width must be at least 1")
    if args.width is not None and args.fez:
        parser.error("--width cannot be used with --fez")
    if args.json and args.show:
        parser.error("--json and --show cannot be used together.")
    if args.json and args.dump:
        parser.error("--json and --dump cannot be used together.")
    if args.json and args.cost and args.run:
        parser.error("--json, --cost and --run cannot all be used together.")

    # Collect (filename_or_None, qasm_code) pairs.
    if args.qasm_files:
        inputs = []
        for path in args.qasm_files:
            with open(path, "r", encoding="utf-8") as f:
                inputs.append((path, f.read()))
    else:
        inputs = [(None, sys.stdin.read())]

    multiple = len(inputs) > 1

    basis_kwargs = dict(clifford_t=args.clifford_t, cx_u=args.cx_u, cx_sx=args.cx_sx, ecr_sx=args.ecr_sx, cz_sx=args.cz_sx, iswap_rx=args.iswap_rx, rzz_rx=args.rzz_rx, rxx_rx=args.rxx_rx, xxphase_rx=args.xxphase_rx, quantinuum_h=args.quantinuum_h, ionq_aria=args.ionq_aria, ionq_forte=args.ionq_forte, syc_phxz=args.google_sycamore, sqrtiswap_phxz=args.sqrtiswap_phxz or args.google_sqrtiswap, fez=args.fez, ibm_eagle=args.ibm_eagle, ibm_heron=args.ibm_heron, ibm_heron_frac=args.ibm_heron_frac, rigetti_ankaa=args.rigetti_ankaa)

    json_results = [] if args.json and multiple else None

    for file_idx, (filename, qasm3_code) in enumerate(inputs):
        if not qasm3_code.strip():
            source = filename or "stdin"
            raise SystemExit(f"No OpenQASM code in {source}.")

        if qasm3_code.lstrip().startswith("OPENQASM 2"):
            from qiskit import qasm2 as _qasm2_mod
            qc = _qasm2_mod.loads(qasm3_code)
        else:
            qc = parse(qasm3_code)

        if args.width is not None:
            if qc.num_qubits > args.width:
                source = filename or "stdin"
                raise SystemExit(
                    f"{source}: circuit has {qc.num_qubits} qubits, which exceeds --width {args.width}."
                )
            num_ancillas = args.width - qc.num_qubits
        else:
            num_ancillas = 0

        if num_ancillas > 0:
            from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
            hls_config = HLSConfig(mcx=[("default", {"num_clean_ancillas": num_ancillas})])
        else:
            hls_config = None

        _cirq_qasm2 = None
        _pytket_qasm2 = None
        _compiled_cirq = None

        if args.fez:
            backend = FakeFez()
            pm = generate_preset_pass_manager(
                optimization_level=args.opt_level,
                backend=backend,
                seed_transpiler=777,
            )
            selected = pm.run(qc)
        elif args.clifford_t:
            pm_kwargs = dict(
                optimization_level=args.opt_level,
                basis_gates=CLIFFORD_T_BASIS,
                seed_transpiler=777,
            )
            if args.eps is not None:
                pm_kwargs["unitary_synthesis_method"] = "gridsynth"
                pm_kwargs["unitary_synthesis_plugin_config"] = {"epsilon": args.eps}
            if hls_config is not None:
                pm_kwargs["hls_config"] = hls_config
            pm = generate_preset_pass_manager(**pm_kwargs)
            selected = pm.run(qc)
        elif args.cx_u or args.cx_sx or args.ecr_sx or args.cz_sx or args.iswap_rx or args.rzz_rx or args.rxx_rx:
            basis = (
                CX_U_BASIS if args.cx_u else
                CX_SX_BASIS if args.cx_sx else
                ECR_SX_BASIS if args.ecr_sx else
                CZ_SX_BASIS if args.cz_sx else
                ISWAP_RX_BASIS if args.iswap_rx else
                RZZ_RX_BASIS if args.rzz_rx else
                RXX_RX_BASIS
            )
            pm = generate_preset_pass_manager(
                optimization_level=args.opt_level,
                basis_gates=basis,
                seed_transpiler=777,
                **({"hls_config": hls_config} if hls_config is not None else {}),
            )
            selected = pm.run(qc)
        elif args.xxphase_rx:
            try:
                from pytket.passes import AutoRebase
                from pytket import OpType
            except ImportError:
                raise SystemExit("--xy-rx requires pytket: pip install pytket")
            rebase = AutoRebase({OpType.XXPhase, OpType.Rx, OpType.Rz})
            _tk_circuit, selected, _pytket_qasm2 = _compile_pytket(
                qc, rebase_pass=rebase, gateset_name="xy-rx"
            )
        elif args.quantinuum_h:
            try:
                from pytket.passes import AutoRebase
                from pytket import OpType
            except ImportError:
                raise SystemExit("--quantinuum-h requires pytket: pip install pytket")
            rebase = AutoRebase({OpType.PhasedX, OpType.ZZPhase, OpType.Rz})
            _tk_circuit, selected, _pytket_qasm2 = _compile_pytket(
                qc, rebase_pass=rebase, gateset_name="quantinuum-h"
            )
        elif args.sqrtiswap_phxz or args.google_sycamore or args.google_sqrtiswap:
            gateset_name = "syc" if args.google_sycamore else "sqrtiswap"
            _compiled_cirq, selected, _cirq_qasm2 = _compile_cirq(qc, gateset_name=gateset_name)
        elif args.ibm_eagle or args.ibm_heron or args.ibm_heron_frac or args.rigetti_ankaa:
            basis = (
                IBM_EAGLE_BASIS if args.ibm_eagle else
                IBM_HERON_BASIS if args.ibm_heron else
                IBM_HERON_FRAC_BASIS if args.ibm_heron_frac else
                RIGETTI_ANKAA_BASIS
            )
            pm = generate_preset_pass_manager(
                optimization_level=args.opt_level,
                basis_gates=basis,
                seed_transpiler=777,
                **({"hls_config": hls_config} if hls_config is not None else {}),
            )
            selected = pm.run(qc)
        elif args.ionq_aria or args.ionq_forte:
            gateset_name = "aria" if args.ionq_aria else "forte"
            _compiled_cirq, selected, _cirq_qasm2 = _compile_ionq(qc, gateset_name=gateset_name)
        else:
            selected = qc

        # With multiple files, default to --cost rather than --show.
        explicitly_selected_output = args.show or args.dump or args.dump2 or args.run or args.cost or args.json
        do_show = args.show or (not explicitly_selected_output and not multiple)
        do_dump = args.dump
        do_dump2 = args.dump2
        do_run = args.run
        default_cost = do_show or (multiple and not explicitly_selected_output)
        do_cost = (args.cost or (default_cost and (args.show or not args.run))) and not args.no_cost
        do_json = args.json

        need_blank = False

        if multiple and not do_json:
            if file_idx > 0:
                print()
            print(f"### {filename}")
            need_blank = True

        if do_show:
            if need_blank:
                print()
            diagram = str(selected.draw(fold=args.fold)).replace("|0>", "|0⟩")
            if _cirq_qasm2 is not None:
                for name in ("Phxz", "Syc", "Sqrt_iswap", "Gpi2", "Gpi", "Ms", "Zz"):
                    diagram = diagram.replace(name, name.lower())
            print(diagram)
            need_blank = True

        if do_dump:
            if need_blank:
                print()
            native_qasm2 = _cirq_qasm2 or _pytket_qasm2
            if native_qasm2 is not None:
                print(native_qasm2.replace(
                    'OPENQASM 2.0;\ninclude "qelib1.inc";',
                    'OPENQASM 3.0;\ninclude "stdgates.inc";',
                    1,
                ))
            else:
                from qiskit import qasm3
                print(qasm3.dumps(selected))
            need_blank = True

        if do_dump2:
            from qiskit import qasm2
            if need_blank:
                print()
            native_qasm2 = _cirq_qasm2 or _pytket_qasm2
            qasm2_out = native_qasm2 if native_qasm2 is not None else qasm2.dumps(selected)
            print(re.sub(r'\bp\(', 'rz(', qasm2_out))
            need_blank = True

        if do_cost:
            if need_blank:
                print()
            print_costs(selected, **basis_kwargs)
            need_blank = True

        if do_json and not args.run:
            result = collect_costs(selected, **basis_kwargs)
            if multiple:
                result["file"] = filename
                json_results.append(result)
            else:
                import json
                print(json.dumps(result, indent=2))

        if do_run:
            if do_json:
                clbits, values = run_circuit_and_capture(selected)
                result = {
                    "clbits": format_clbits(clbits) or None,
                    "statevector": [
                        {"basis": basis, "re": val.real, "im": val.imag}
                        for basis, val in values
                    ],
                }
                if multiple:
                    result["file"] = filename
                    json_results.append(result)
                else:
                    import json
                    print(json.dumps(result, indent=2))
            else:
                if need_blank:
                    print()
                clbits, values = run_circuit_and_capture(selected)
                clbit_str = format_clbits(clbits)
                if clbit_str:
                    print(f"classical bits: {clbit_str}")
                if values:
                    max_abs = max(abs(v) for _, v in values)
                    print_pretty_state(values, max_abs=max_abs)
                else:
                    print("0")

    if json_results is not None:
        import json
        print(json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()
