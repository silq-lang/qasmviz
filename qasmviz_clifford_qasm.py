"""OpenQASM definitions and phase-preserving export for named Cliffords."""

import copy
from functools import lru_cache

import openqasm3
from openqasm3 import ast
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import ControlFlowOp

from qasmviz_basis import clifford_decomposition


# These Qiskit Clifford names are not supplied by OpenQASM's stdgates.inc.
NONSTANDARD_CLIFFORDS = frozenset(('sxdg', 'iswap', 'ecr', 'dcx'))


@lru_cache(maxsize=4)
def _definition(name):
    circuit = clifford_decomposition(name, ('h', 's', 'cx'))
    qubits = [ast.Identifier(f'q{i}') for i in range(circuit.num_qubits)]
    body = []
    if float(circuit.global_phase):
        body.append(ast.QuantumPhase([], ast.FloatLiteral(float(circuit.global_phase)), []))
    for instruction in circuit.data:
        body.append(ast.QuantumGate([], ast.Identifier(instruction.operation.name), [],
                                    [qubits[circuit.find_bit(q).index] for q in instruction.qubits]))
    return ast.QuantumGateDefinition(ast.Identifier(name), [], qubits, body)


def clifford_definitions(names):
    # A definition describes a native basis gate; its body uses stdgates.inc
    # solely to give that gate a portable and phase-correct QASM meaning.
    return [copy.deepcopy(_definition(name)) for name in sorted(set(names) & NONSTANDARD_CLIFFORDS)]


@lru_cache(maxsize=1)
def _exporter_preserves_phase():
    probe = QuantumCircuit(1, global_phase=0.7)
    return any(isinstance(s, ast.QuantumPhase) for s in openqasm3.parse(qasm3.dumps(probe)).statements)


def _add_phases(statements, circuit):
    """Restore phases dropped by Qiskit 2.5's QASM exporter, also in blocks."""
    controls = [s for s in statements if isinstance(s, (ast.BranchingStatement, ast.ForInLoop,
                                                       ast.WhileLoop, ast.SwitchStatement))]
    operations = [i.operation for i in circuit.data if isinstance(i.operation, ControlFlowOp)]
    if len(controls) != len(operations):
        raise ValueError('cannot match exported control flow when preserving Clifford+T phases')
    for statement, operation in zip(controls, operations):
        if isinstance(statement, ast.BranchingStatement):
            blocks = [statement.if_block, statement.else_block][:len(operation.blocks)]
        elif isinstance(statement, (ast.ForInLoop, ast.WhileLoop)):
            blocks = [statement.block]
        else:
            blocks = [case.statements for _, case in statement.cases]
            if statement.default is not None:
                blocks.append(statement.default.statements)
        if len(blocks) != len(operation.blocks):
            raise ValueError('cannot match exported blocks when preserving Clifford+T phases')
        for body, block in zip(blocks, operation.blocks):
            _add_phases(body, block)
    if float(circuit.global_phase):
        # Definitions and declarations remain before the executable phase.
        position = next((i for i, s in enumerate(statements) if not isinstance(
            s, (ast.Include, ast.QuantumGateDefinition, ast.QubitDeclaration, ast.ClassicalDeclaration))), len(statements))
        statements.insert(position, ast.QuantumPhase([], ast.FloatLiteral(float(circuit.global_phase)), []))


def dumps_clifford_t(circuit):
    program = openqasm3.parse(qasm3.dumps(circuit))
    for index, statement in enumerate(program.statements):
        if isinstance(statement, ast.QuantumGateDefinition) and statement.name.name in NONSTANDARD_CLIFFORDS:
            program.statements[index] = copy.deepcopy(_definition(statement.name.name))
    if not _exporter_preserves_phase():
        _add_phases(program.statements, circuit)
    return openqasm3.dumps(program)
