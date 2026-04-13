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
    respect_barriers: bool = True,
) -> tuple[int, int]:
    """
    Dependency-aware metric depth/count on the circuit DAG.

    The metric depth is the length of the longest dependency-respecting path,
    counting:
      - 1 for each interesting node
      - additionally 1 for each barrier node if respect_barriers=True

    Non-interesting, non-barrier nodes propagate the current metric layer
    without increasing it.
    """
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

        here = base + (1 if interesting or is_barrier else 0)
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

    Current semantics:
      - only measurements that write clbits later consumed by If/While/Switch
        conditions are considered
      - a later overwrite kills the earlier measurement
      - only control-flow ops whose blocks contain quantum effects count
      - analysis is over top-level measurement definitions; nested blocks are
        scanned for control uses, but measurements inside nested blocks are not
        returned by this function
    """
    from qiskit.circuit import ControlFlowOp, IfElseOp, WhileLoopOp, SwitchCaseOp

    data = list(circuit.data)
    active: set[int] = set()

    # Last top-level measurement that defined each clbit.
    last_measure_def: dict[object, int] = {}

    def process_block(block, reaching_defs):
        current_defs = dict(reaching_defs)

        for instruction in block.data:
            op = instruction.operation

            if op.name == "measure":
                for cbit in instruction.clbits:
                    current_defs[cbit] = -1  # nested measurement, not top-level
                continue

            if isinstance(op, (IfElseOp, WhileLoopOp)):
                used = set(_iter_condition_clbits(op.condition))
                if _block_has_quantum_effects(block) and used:
                    for cbit in used:
                        src = current_defs.get(cbit)
                        if src is not None and src >= 0:
                            active.add(src)

                for inner in op.blocks:
                    process_block(inner, current_defs)
                continue

            if isinstance(op, SwitchCaseOp):
                used = set(_iter_condition_clbits(op.target))
                if _block_has_quantum_effects(block) and used:
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

        return

    # Top-level pass.
    current_defs = dict(last_measure_def)
    for i, instruction in enumerate(data):
        op = instruction.operation

        if op.name == "measure":
            for cbit in instruction.clbits:
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

def mcm_count(circuit) -> int:
    return len(active_measurement_indices(circuit))

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
        ("qubits", circuit.num_qubits),
        ("gates", format_gate_counts(circuit)),
    ]

    if clifford_t:
        tc, td = t_metrics(circuit)
        if tc:
            rows.append(("t-count", tc))
            rows.append(("t-depth", td))

    if clifford_t or cx1q:
        cxc, cxd = cx_metrics(circuit)
        if cxc:
            rows.append(("cx-count", cxc))
            rows.append(("cx-depth", cxd))

    mcmc = mcm_count(circuit)
    if mcmc:
        rows.append(("mcm-count", mcmc))

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
