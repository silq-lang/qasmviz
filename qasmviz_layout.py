"""Materialize transpiler permutations before drawing, exporting or running."""

from collections import deque
from functools import lru_cache

from qiskit import QuantumCircuit
from qiskit.transpiler import TranspilerError, generate_preset_pass_manager


def restore_layout(circuit, *, basis_gates=None, target=None, initially_zero=False):
    """Return a circuit whose output wires have the original logical order.

    With a physical target, route the restoring swaps over its actual edges.
    Otherwise the initial layout can be undone by relabelling circuit wires.
    Keep all wires, including idle wires and backend-allocated ancillas.
    """
    if circuit.layout is None:
        return circuit
    n = circuit.num_qubits
    initial = circuit.layout.initial_index_layout(filter_ancillas=False)
    final = circuit.layout.final_index_layout(filter_ancillas=False)
    if any(len(p) != n or set(p) != set(range(n)) for p in (initial, final)):
        raise TranspilerError('unsupported transpiler layout')
    inverse = [0] * n
    for logical, physical in enumerate(initial):
        inverse[physical] = logical

    result = circuit.copy_empty_like()
    # Clearing this metadata also makes QASM emit declarations for idle wires,
    # instead of only the physical $indices mentioned by executable gates.
    result._layout = None
    adjacency = None
    if target is not None:
        coupling = target.build_coupling_map()
        if coupling is not None:
            adjacency = [set() for _ in range(n)]
            for a, b in coupling.get_edges():
                adjacency[a].add(b)
                adjacency[b].add(a)

    @lru_cache(maxsize=None)
    def swap(a, b):
        if target is None:
            local = QuantumCircuit(2)
            local.swap(0, 1)
            if basis_gates is None or 'swap' in basis_gates:
                return local
            from qasmviz_basis import EXTENDED_CLIFFORD_T_BASIS, clifford_decomposition
            if set(basis_gates) <= set(EXTENDED_CLIFFORD_T_BASIS):
                return clifford_decomposition('swap', basis_gates)
            return generate_preset_pass_manager(
                0, basis_gates=list(basis_gates), qubits_initially_zero=False,
            ).run(local)
        physical = QuantumCircuit(n)
        physical.swap(a, b)
        compiled = generate_preset_pass_manager(
            0, target=target, initial_layout=list(range(n)), routing_method='none',
            qubits_initially_zero=False,
        ).run(physical)
        if compiled.layout.final_index_layout(filter_ancillas=False) != list(range(n)):
            raise TranspilerError('restoring SWAP introduced another layout permutation')
        local = QuantumCircuit(2, global_phase=compiled.global_phase)
        indices = {a: 0, b: 1}
        for instruction in compiled.data:
            wires = [compiled.find_bit(q).index for q in instruction.qubits]
            if instruction.clbits or any(q not in indices for q in wires):
                raise TranspilerError('restoring SWAP used unexpected wires')
            local.append(instruction.operation, [indices[q] for q in wires])
        return local

    def exchange(a, b):
        # Swapping along a path and then back (except its last edge) exchanges
        # just its endpoints. Intermediate states return to their own wires.
        path = [a, b]
        if adjacency is not None:
            parents, pending = {a: None}, deque([a])
            while pending and b not in parents:
                node = pending.popleft()
                for neighbor in sorted(adjacency[node]):
                    if neighbor not in parents:
                        parents[neighbor] = node
                        pending.append(neighbor)
            if b not in parents:
                raise TranspilerError('cannot restore output wires across disconnected target components')
            path = [b]
            while path[-1] != a:
                path.append(parents[path[-1]])
            path.reverse()
        edges = list(zip(path, path[1:]))
        for left, right in edges + edges[-2::-1]:
            result.compose(swap(left, right), qubits=[left, right], inplace=True)

    def restore(positions):
        occupants = [0] * n
        for logical, position in enumerate(positions):
            occupants[position] = logical
        for logical in range(n):
            position = positions[logical]
            if position == logical:
                continue
            exchange(logical, position)
            displaced = occupants[logical]
            positions[logical], positions[displaced] = logical, position
            occupants[logical], occupants[position] = logical, displaced

    if target is not None and not initially_zero:
        restore(inverse.copy())
    wires = inverse if target is None else list(range(n))
    for instruction in circuit.data:
        result.append(instruction.operation,
                      [result.qubits[wires[circuit.find_bit(q).index]] for q in instruction.qubits],
                      [result.clbits[circuit.find_bit(c).index] for c in instruction.clbits])
    restore([wires[physical] for physical in final])
    return result
