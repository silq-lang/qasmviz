import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*You have imported samplomatic==.*beta development.*",
    category=UserWarning,
    module=r"samplomatic(\..*)?",
)

import sys
import math
import cmath
import argparse

from qiskit_qasm3_import import parse
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import qblaze
import qblaze.qiskit


CLIFFORD_T_BASIS = ["cx", "h", "s", "sdg", "t", "tdg"]
CX_1Q_BASIS = ["cx", "u"]


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


# Single-qubit gates with a free continuous angle parameter — the gates that
# drive Clifford+T / gridsynth synthesis cost.
_ROTATION_GATES = {"rx", "ry", "rz", "p", "u1", "u2", "u3", "u"}


def _dyadic_t_cost(angle: float) -> int | None:
    """
    Return the exact T-count required to synthesize a rotation by ``angle``
    radians, or None if the angle is not a dyadic rational multiple of π.

    A rotation by kπ/2^n (in lowest terms, k odd, n ≥ 0) costs max(0, n-1)
    T-gates:
      - n=0: multiple of π      → identity or Z/X up to Clifford → 0
      - n=1: multiple of π/2    → S/Sdg or X/Y up to Clifford   → 0
      - n=2: multiple of π/4    → T/Tdg up to Clifford           → 1
      - n=k: multiple of π/2^k                                   → k-1

    Returns None for angles that are not dyadic rational multiples of π
    (i.e. require approximate synthesis).
    """
    # Normalise to [0, 2π).
    turns = (angle / math.pi) % 2.0   # angle in units of π, mod 2

    # Check if turns is close to a dyadic rational k/2^n for increasing n.
    MAX_N = 20
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
            return max(0, n - 1)
    return None  # not a dyadic rational multiple of π


def _gate_t_cost(op) -> int | None:
    """
    Return the T-cost of a rotation gate operation.

    For gates with multiple angle parameters (u, u2, u3), the cost is the
    maximum over all parameters, since each non-Clifford parameter requires
    independent synthesis but they can be parallelised.

    Returns None if any parameter requires approximate synthesis.
    """
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
    return max(costs, default=0)


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

    depth, count = metric_depth_and_count(
        circuit,
        is_interesting=is_rotation_collect,
        respect_barriers=True,
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
        )
    else:
        t_depth_val = 0  # not meaningful; caller checks n_approx

    return count, depth, t_depth_val, n_approx, breakdown


def format_rotation_breakdown(breakdown: dict[int | None, int]) -> str:
    """
    Format the rotation breakdown as a compact annotation string, e.g.
    ``clifford=2, T=3, T²=1, approx=1``.

    Dyadic orders > 1 are written as T^n using Unicode superscripts.
    """
    _super = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    def superscript(n: int) -> str:
        return str(n).translate(_super)

    parts = []
    if breakdown.get(0, 0):
        parts.append(f"clifford={breakdown[0]}")
    # Emit dyadic orders in ascending order.
    for n in sorted(k for k in breakdown if isinstance(k, int) and k >= 1):
        label = "T" if n == 1 else f"T{superscript(n)}"
        parts.append(f"{label}={breakdown[n]}")
    if breakdown.get(None, 0):
        parts.append(f"approx={breakdown[None]}")
    return "  (" + ", ".join(parts) + ")" if parts else ""

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

def format_gate_counts(circuit) -> str:
    counts = dict(circuit.count_ops())

    preferred_order = [
        "cx",
        "cz",
        "swap",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "x",
        "y",
        "z",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "p",
        "u",
        "u1",
        "u2",
        "u3",
        "id",
        "reset",
        "measure",
        "barrier",
    ]
    rank = {name: i for i, name in enumerate(preferred_order)}

    items = sorted(
        counts.items(),
        key=lambda kv: (rank.get(kv[0], len(preferred_order)), kv[0]),
    )
    return ", ".join(f"{name}={count}" for name, count in items)

def print_costs(circuit, *, clifford_t: bool, cx1q: bool) -> None:
    rows: list[tuple[str, object]] = [
        ("width", circuit.num_qubits),
        ("depth", circuit.depth()),
        ("gates", format_gate_counts(circuit)),
    ]

    if clifford_t:
        tc, td = t_metrics(circuit)
        if tc:
            rows.append(("t-count", tc))
            rows.append(("t-depth", td))

    # Report CX-count/depth or two-qubit-count/depth, but not both:
    #   - if CX is the only two-qubit gate present, CX metrics are sufficient
    #   - if other two-qubit gates are present, CX metrics are misleading and
    #     two-qubit metrics capture the full picture
    #   - if any 3+-qubit gates are present, two-qubit metrics are incomplete
    #     and neither CX nor 2q metrics are reported (unless basis-translated)
    twoq_names = two_qubit_gate_names(circuit)
    if clifford_t or cx1q:
        # After basis translation these modes guarantee only CX as the
        # two-qubit gate, so CX metrics are always the right choice.
        cxc, cxd = cx_metrics(circuit)
        if cxc:
            rows.append(("cx-count", cxc))
            rows.append(("cx-depth", cxd))
    elif has_many_qubit_gates(circuit):
        pass  # 2q metrics would be incomplete; omit entirely
    elif twoq_names == {"cx"}:
        cxc, cxd = cx_metrics(circuit)
        if cxc:
            rows.append(("cx-count", cxc))
            rows.append(("cx-depth", cxd))
    elif twoq_names:
        tqc, tqd = two_qubit_metrics(circuit)
        if tqc:
            rows.append(("2q-count", tqc))
            rows.append(("2q-depth", tqd))

    rc, rd, rt, rn_approx, rbreakdown = rotation_metrics(circuit)
    if rc:
        rows.append(("rot-count", f"{rc}{format_rotation_breakdown(rbreakdown)}"))
        if rn_approx == 0:
            t_depth_annotation = f"  (T-depth: {rt})"
        else:
            t_depth_annotation = "  (T-depth: n/a)"
        rows.append(("rot-depth", f"{rd}{t_depth_annotation}"))

    mcmc, mcmd = mcm_metrics(circuit)
    if mcmc:
        rows.append(("mcm-count", mcmc))
        rows.append(("mcm-depth", mcmd))

    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label.rjust(width)}: {value}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read OpenQASM 3 from a file or stdin, optionally dump the selected circuit, print costs, and/or run it with qblaze."
    )
    parser.add_argument(
        "qasm_file",
        nargs="?",
        help="Optional .qasm file to read. Reads from stdin if omitted.",
    )

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--fez",
        action="store_true",
        help="Transpile for FakeFez and use the execution-ready circuit.",
    )
    compile_group.add_argument(
        "--clifford-t",
        action="store_true",
        dest="clifford_t",
        help="Transpile into the Clifford+T basis {cx, h, s, sdg, t, tdg}.",
    )
    compile_group.add_argument(
        "--cx1q",
        action="store_true",
        help="Transpile into the basis {cx, u}.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        help="Maximum allowed approximation error for Clifford+T synthesis. Requires --clifford-t.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print the selected circuit.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Simulate the selected circuit with qblaze and print classical bits and final sparse state.",
    )
    parser.add_argument(
        "--cost",
        action="store_true",
        help="Print operation counts even when not dumping the circuit.",
    )
    parser.add_argument(
        "--no-cost",
        action="store_true",
        help="Do not print operation counts, including after --dump.",
    )
    parser.add_argument(
        "--fold",
        nargs="?",
        type=int,
        const=None,
        default=-1,
        help=(
            "Circuit draw fold width. Default: -1 (no folding). "
            "Use bare --fold for automatic width detection."
        ),
    )
    args = parser.parse_args()

    if args.eps is not None and not args.clifford_t:
        parser.error("--eps requires --clifford-t")
    if args.eps is not None and args.eps <= 0:
        parser.error("--eps must be positive")

    if args.qasm_file is not None:
        with open(args.qasm_file, "r", encoding="utf-8") as f:
            qasm3_code = f.read()
    else:
        qasm3_code = sys.stdin.read()

    if not qasm3_code.strip():
        raise SystemExit("No OpenQASM 3 code was provided.")

    qc = parse(qasm3_code)

    if args.fez:
        backend = FakeFez()
        pm = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            seed_transpiler=777,
        )
        selected = pm.run(qc)
    elif args.clifford_t:
        pm_kwargs = dict(
            optimization_level=3,
            basis_gates=CLIFFORD_T_BASIS,
            seed_transpiler=777,
        )
        if args.eps is not None:
            pm_kwargs["unitary_synthesis_method"] = "gridsynth"
            pm_kwargs["unitary_synthesis_plugin_config"] = {"epsilon": args.eps}
        pm = generate_preset_pass_manager(**pm_kwargs)
        selected = pm.run(qc)
    elif args.cx1q:
        pm = generate_preset_pass_manager(
            optimization_level=3,
            basis_gates=CX_1Q_BASIS,
            seed_transpiler=777,
        )
        selected = pm.run(qc)
    else:
        selected = qc

    explicitly_selected_output = args.dump or args.run or args.cost

    do_dump = args.dump or not explicitly_selected_output
    do_run = args.run
    do_cost = (do_dump or args.cost) and not args.no_cost

    need_blank = False

    if do_dump:
        print(selected.draw(fold=args.fold))
        need_blank = True

    if do_cost:
        if need_blank:
            print()
        print_costs(selected, clifford_t=args.clifford_t, cx1q=args.cx1q)
        need_blank = True

    if do_run:
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


if __name__ == "__main__":
    main()
