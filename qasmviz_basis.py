"""Exact validation and translation of named Clifford generator sets.

Completeness is checked modulo global phase, including Pauli signs. The
single-qubit generators must reach all 24 Cliffords without an ancillary
qubit, and their two-qubit placements must reach CX. H, S and CX then give
the Clifford group on any number of qubits. Translation restores the global
phase omitted by the tableau representation.
"""

from collections import deque
from functools import lru_cache
from itertools import permutations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, Gate
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Clifford, Operator, PauliList, get_clifford_gate_names
from qiskit.transpiler import TransformationPass, TranspilerError


CLIFFORD_GATES = tuple(get_clifford_gate_names())
EXTENDED_CLIFFORD_T_BASIS = (*CLIFFORD_GATES, 't', 'tdg')
_GATES = {name: gate for name, gate in get_standard_gate_name_mapping().items() if name in CLIFFORD_GATES}


def _signature(clifford):
    n = clifford.num_qubits
    return tuple(sum(int(x) << i for i, x in enumerate(row_x))
                 | sum(int(z) << (n + i) for i, z in enumerate(row_z))
                 | (int(sign) << (2 * n))
                 for row_x, row_z, sign in zip(clifford.x, clifford.z, clifford.phase))


@lru_cache(maxsize=64)
def _action(name, wires, n):
    """Map all signed Hermitian Paulis under conjugation by one gate."""
    circuit = QuantumCircuit(n)
    circuit.append(_GATES[name], wires)
    labels = np.arange(1 << (2 * n))[:, None]
    x = (labels >> np.arange(n)) & 1
    z = (labels >> (n + np.arange(n))) & 1
    paulis = PauliList.from_symplectic(z.astype(bool), x.astype(bool))
    images = paulis.evolve(Clifford(circuit), frame='s')
    assert not np.any(images.phase % 2)
    positive = tuple(sum(int(bit) << i for i, bit in enumerate(row_x))
                     | sum(int(bit) << (n + i) for i, bit in enumerate(row_z))
                     | (int(phase // 2) << (2 * n))
                     for row_x, row_z, phase in zip(images.x, images.z, images.phase))
    return positive + tuple(code ^ (1 << (2 * n)) for code in positive)


def _closure(names, n, targets=None):
    # Use integer tuples instead of constructing a Clifford object at each BFS
    # step. Deduplicate generators with the same action (e.g. swapped CZ).
    actions = {}
    for name in names:
        for wires in permutations(range(n), _GATES[name].num_qubits):
            actions.setdefault(_action(name, wires, n), (name, wires))
    identity = tuple(1 << i for i in range(2 * n))
    parents = {identity: None}
    pending = deque([identity])
    missing = None if targets is None else set(targets) - {identity}
    while pending and (missing is None or missing):
        before = pending.popleft()
        for action, instruction in actions.items():
            after = tuple(action[pauli] for pauli in before)
            if after in parents:
                continue
            parents[after] = before, instruction
            pending.append(after)
            if missing is not None:
                missing.discard(after)
    return parents


def _with_phase(circuit, gate):
    matrix = Operator(circuit).data
    target = gate.to_matrix()
    phase = np.angle(np.vdot(matrix, target))
    circuit.global_phase += float(phase)
    if not np.allclose(Operator(circuit).data, target, atol=1e-12, rtol=0):
        raise RuntimeError(f'inconsistent Clifford decomposition for {gate.name!r}')
    return circuit


def _word(gate, parents):
    state = _signature(Clifford(gate))
    instructions = []
    while parents[state] is not None:
        state, instruction = parents[state]
        instructions.append(instruction)
    circuit = QuantumCircuit(gate.num_qubits)
    for name, wires in reversed(instructions):
        circuit.append(_GATES[name], wires)
    return _with_phase(circuit, gate)


@lru_cache(maxsize=32)
def _decompositions(names):
    single = tuple(name for name in names if _GATES[name].num_qubits == 1)
    one = _closure(single, 1)
    if len(one) != 24:
        raise ValueError('Clifford generators are incomplete: the one-qubit gates generate '
                         f'{len(one)} of 24 Cliffords; a full one-qubit group is required without ancillas')
    targets = {_signature(Clifford(gate)) for gate in _GATES.values() if gate.num_qubits == 2}
    two = _closure(names, 2, targets)
    if _signature(Clifford(_GATES['cx'])) not in two:
        raise ValueError('Clifford generators are incomplete: they cannot generate CX; '
                         'local Clifford gates and SWAP alone do not generate the full Clifford group')
    return {name: _word(gate, one if gate.num_qubits == 1 else two)
            for name, gate in _GATES.items() if name not in names}


def validate_clifford_basis(gates):
    """Return the complete output basis; T and Tdg are always available."""
    names = tuple(dict.fromkeys(gates))
    unknown = [name for name in names if name not in EXTENDED_CLIFFORD_T_BASIS]
    if unknown:
        raise ValueError('expected fixed Clifford gate names; unsupported generators: '
                         + ', '.join(repr(name) for name in unknown))
    cliffords = tuple(name for name in names if name not in ('t', 'tdg'))
    _decompositions(cliffords)
    return (*cliffords, 't', 'tdg')


def parse_extended_basis(value):
    """Parse [h,s,cx], also accepting quoted gate names and a trailing comma."""
    if value == 'all':
        return validate_clifford_basis(CLIFFORD_GATES)
    value = value.strip()
    if not (value.startswith('[') and value.endswith(']')):
        raise ValueError('--clifford-t-ext expects a bracketed list, for example [h,s,cx]')
    items = value[1:-1].split(',') if value[1:-1].strip() else []
    if items and not items[-1].strip():
        items.pop()
    names = []
    for item in items:
        item = item.strip()
        if len(item) >= 2 and item[0] in ('"', "'") and item[-1] == item[0]:
            item = item[1:-1]
        if not item or not item.isidentifier():
            raise ValueError('--clifford-t-ext expects a list of gate names, for example [h,s,cx]')
        names.append(item)
    return validate_clifford_basis(names)


def clifford_decomposition(name, basis):
    """Return an exact circuit using the selected Clifford gates only."""
    names = tuple(gate for gate in basis if gate not in ('t', 'tdg'))
    if name in names:
        gate = _GATES[name]
        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, range(gate.num_qubits))
        return circuit
    return _decompositions(names)[name].copy()


def restore_logical_wires(circuit, basis):
    """Materialize the layout before exporting or simulating a circuit.

    QASM and qblaze do not interpret Qiskit's final-layout metadata. Restore
    permutations using allowed gates, so displayed costs include that work.
    """
    if circuit.layout is None:
        return circuit
    n = circuit.num_qubits
    initial = circuit.layout.initial_index_layout(filter_ancillas=False)
    final = circuit.layout.final_index_layout(filter_ancillas=False)
    if len(initial) != n or len(final) != n or set(initial) != set(range(n)) or set(final) != set(range(n)):
        raise TranspilerError('unsupported Clifford+T layout')
    physical_to_source = [0] * n
    for logical, physical in enumerate(initial):
        physical_to_source[physical] = logical
    result = circuit.copy_empty_like()
    # copy_empty_like retains this metadata; it would apply the permutation
    # again if left attached to the explicitly rewired circuit.
    result._layout = None
    for instruction in circuit.data:
        result.append(instruction.operation,
                      [result.qubits[physical_to_source[circuit.find_bit(q).index]] for q in instruction.qubits],
                      [result.clbits[circuit.find_bit(c).index] for c in instruction.clbits])
    positions = [physical_to_source[physical] for physical in final]
    occupants = [0] * n
    for logical, position in enumerate(positions):
        occupants[position] = logical
    swap = None
    for logical in range(n):
        position = positions[logical]
        if position == logical:
            continue
        swap = clifford_decomposition('swap', basis) if swap is None else swap
        result.compose(swap, qubits=[logical, position], inplace=True)
        displaced = occupants[logical]
        positions[logical], positions[displaced] = logical, position
        occupants[logical], occupants[position] = logical, displaced
    return result


class CliffordBasisTranslation(TransformationPass):
    """Replace disallowed Cliffords exactly, without introducing T gates."""

    def __init__(self, basis):
        super().__init__()
        self.basis = frozenset(basis)
        names = tuple(name for name in basis if name not in ('t', 'tdg'))
        self.replacements = {name: circuit_to_dag(circuit)
                             for name, circuit in _decompositions(names).items()}

    def run(self, dag):
        for node in list(dag.op_nodes()):
            if isinstance(node.op, ControlFlowOp):
                blocks = [dag_to_circuit(self.run(circuit_to_dag(block))) for block in node.op.blocks]
                dag.substitute_node(node, node.op.replace_blocks(blocks), inplace=True)
            elif node.op.name in self.replacements:
                dag.substitute_node_with_dag(node, self.replacements[node.op.name])
            elif isinstance(node.op, Gate) and node.op.name not in self.basis:
                raise TranspilerError(f'Clifford+T compiler left unsupported gate {node.op.name!r}')
        return dag
