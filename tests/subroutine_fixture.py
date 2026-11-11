"""Small independent interpreter for the subroutine regression fixtures.

It executes actual call frames and Python return exceptions, rather than the
compiler's inlining scheme. Measurement outcomes are supplied by each test;
their probabilities and conditional quantum states are calculated explicitly.
This is deliberately not an OpenQASM implementation for general use.
"""

import math
import operator

import numpy as np
from openqasm3 import ast
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class Frame:
    def __init__(self, parent=None):
        self.parent = parent
        self.values = {}
        self.types = {}

    def owner(self, name):
        if name in self.values:
            return self
        if self.parent is None:
            raise AssertionError(f'unknown fixture variable {name}')
        return self.parent.owner(name)

    def get(self, name):
        return self.owner(name).values[name]


def execute(program, *, initial=None, outcomes=()):
    class Return(Exception):
        def __init__(self, value):
            self.value = value

    class Break(Exception):
        pass

    class Continue(Exception):
        pass

    class End(Exception):
        pass

    global_frame = Frame()
    global_frame.values['pi'] = math.pi
    definitions = {}
    width = 0
    # Fixtures use literal global qubit sizes.
    for statement in program.statements:
        if isinstance(statement, ast.QubitDeclaration):
            size = statement.size.value if statement.size is not None else 1
            global_frame.values[statement.qubit.name] = tuple(range(width, width + size))
            width += size
    state = Statevector(initial if initial is not None else np.eye(1, 2**width, 0).ravel())
    probability = 1.0
    measurements = iter(outcomes)
    steps = 0

    def cast(kind, value, frame):
        if isinstance(kind, ast.BoolType):
            return bool(value)
        if isinstance(kind, ast.FloatType):
            return float(np.float32(value)) if kind.size is not None and kind.size.value == 32 else float(value)
        if isinstance(kind, (ast.IntType, ast.UintType, ast.BitType)):
            return int(value)
        return value

    def selection(index, count, frame):
        if isinstance(index, list):
            assert len(index) == 1
            index = index[0]
        if isinstance(index, ast.RangeDefinition):
            step = value(index.step, frame) if index.step is not None else 1
            start = value(index.start, frame) if index.start is not None else (0 if step > 0 else count - 1)
            end = value(index.end, frame) if index.end is not None else (count - 1 if step > 0 else 0)
            return range(start, end + (1 if step > 0 else -1), step)
        if isinstance(index, ast.DiscreteSet):
            return [value(v, frame) for v in index.values]
        return [value(index, frame)]

    def measure(wire):
        nonlocal state, probability
        outcome = next(measurements)
        mask = (np.arange(len(state.data)) >> wire) & 1
        amplitudes = np.where(mask == outcome, state.data, 0)
        p = float(np.linalg.norm(amplitudes)**2)
        probability *= p
        if p > 1e-15:
            state = Statevector(amplitudes / np.sqrt(p))
        return outcome

    def value(node, frame):
        if isinstance(node, (ast.IntegerLiteral, ast.FloatLiteral, ast.BooleanLiteral)):
            return node.value
        if isinstance(node, ast.Identifier):
            return frame.get(node.name)
        if isinstance(node, ast.ArrayLiteral):
            return [value(item, frame) for item in node.values]
        if isinstance(node, ast.IndexedIdentifier):
            result = frame.get(node.name.name)
            for index in node.indices:
                result = tuple(result[i] for i in selection(index, len(result), frame))
            return result
        if isinstance(node, ast.IndexExpression):
            collection = value(node.collection, frame)
            if isinstance(collection, list):
                index = node.index
                assert isinstance(index, list) and len(index) == 1
                return collection[value(index[0], frame)]
            return tuple(collection[i] for i in selection(node.index, len(collection), frame))
        if isinstance(node, ast.Concatenation):
            return value(node.lhs, frame) + value(node.rhs, frame)
        if isinstance(node, ast.Cast):
            return cast(node.type, value(node.argument, frame), frame)
        if isinstance(node, ast.UnaryExpression):
            return {'!': operator.not_, '-': operator.neg, '~': operator.invert}[node.op.name](value(node.expression, frame))
        if isinstance(node, ast.BinaryExpression):
            a, b = value(node.lhs, frame), value(node.rhs, frame)
            if node.op.name == '/':
                return int(a / b) if isinstance(a, int) and isinstance(b, int) else a / b
            return {'+': operator.add, '-': operator.sub, '*': operator.mul, '%': operator.mod,
                    '==': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le,
                    '>': operator.gt, '>=': operator.ge, '&&': lambda a, b: a and b,
                    '||': lambda a, b: a or b}[node.op.name](a, b)
        if isinstance(node, ast.QuantumMeasurement):
            return sum(measure(wire) << i for i, wire in enumerate(value(node.qubit, frame)))
        if isinstance(node, ast.FunctionCall):
            definition, defining_frame = definitions[node.name.name]
            arguments = [value(a, frame) for a in node.arguments]
            call_frame = Frame(defining_frame)
            for formal, actual in zip(definition.arguments, arguments):
                if isinstance(formal, ast.ClassicalArgument):
                    call_frame.types[formal.name.name] = formal.type
                    actual = cast(formal.type, actual, frame)
                call_frame.values[formal.name.name] = actual
            try:
                block(definition.body, call_frame)
            except Return as returned:
                return cast(definition.return_type, returned.value, call_frame)
            return None
        raise AssertionError(f'unexpected fixture expression: {node}')

    def store(target, rhs, frame, op='='):
        if isinstance(target, ast.IndexedIdentifier):
            collection = frame.get(target.name.name)
            assert isinstance(collection, list) and len(target.indices) == 1
            index = value(target.indices[0][0], frame)
            if op != '=':
                rhs = {'+=': operator.add, '-=': operator.sub}[op](collection[index], rhs)
            collection[index] = rhs
            return
        assert isinstance(target, ast.Identifier), target
        owner = frame.owner(target.name)
        if op != '=':
            rhs = {'+=': operator.add, '-=': operator.sub}[op](owner.values[target.name], rhs)
        owner.values[target.name] = cast(owner.types.get(target.name), rhs, frame)

    def block(statements, frame):
        nonlocal state, steps
        for statement in statements:
            steps += 1
            assert steps < 10000, 'fixture failed to terminate'
            if isinstance(statement, (ast.ClassicalDeclaration, ast.ConstantDeclaration)):
                rhs = value(statement.init_expression, frame) if statement.init_expression is not None else 0
                frame.types[statement.identifier.name] = statement.type
                frame.values[statement.identifier.name] = cast(statement.type, rhs, frame)
            elif isinstance(statement, ast.AliasStatement):
                frame.values[statement.target.name] = value(statement.value, frame)
            elif isinstance(statement, ast.SubroutineDefinition):
                definitions[statement.name.name] = (statement, frame)
            elif isinstance(statement, ast.ExpressionStatement):
                value(statement.expression, frame)
            elif isinstance(statement, ast.ClassicalAssignment):
                store(statement.lvalue, value(statement.rvalue, frame), frame, statement.op.name)
            elif isinstance(statement, ast.QuantumMeasurementStatement):
                measured = value(statement.measure, frame)
                if statement.target is not None:
                    store(statement.target, measured, frame)
            elif isinstance(statement, ast.QuantumReset):
                for wire in value(statement.qubits, frame):
                    p1 = sum(abs(a)**2 for i, a in enumerate(state.data) if (i >> wire) & 1)
                    # These fixtures reset already measured qubits.
                    assert p1 < 1e-10 or abs(p1 - 1) < 1e-10
                    if p1 > 0.5:
                        circuit = QuantumCircuit(width)
                        circuit.x(wire)
                        state = state.evolve(circuit)
            elif isinstance(statement, ast.QuantumGate):
                circuit = QuantumCircuit(width)
                operands = [value(q, frame) for q in statement.qubits]
                count = max(map(len, operands))
                for i in range(count):
                    getattr(circuit, statement.name.name)(*(value(a, frame) for a in statement.arguments),
                                                         *(q[0 if len(q) == 1 else i] for q in operands))
                state = state.evolve(circuit)
            elif isinstance(statement, ast.QuantumPhase):
                state = Statevector(state.data * np.exp(1j * value(statement.argument, frame)))
            elif isinstance(statement, ast.BranchingStatement):
                block(statement.if_block if value(statement.condition, frame) else statement.else_block, Frame(frame))
            elif isinstance(statement, ast.CompoundStatement):
                block(statement.statements, Frame(frame))
            elif isinstance(statement, ast.ForInLoop):
                for item in selection(statement.set_declaration, 0, frame):
                    loop_frame = Frame(frame)
                    loop_frame.values[statement.identifier.name] = item
                    loop_frame.types[statement.identifier.name] = statement.type
                    try:
                        block(statement.block, loop_frame)
                    except Continue:
                        continue
                    except Break:
                        break
            elif isinstance(statement, ast.WhileLoop):
                while value(statement.while_condition, frame):
                    try:
                        block(statement.block, Frame(frame))
                    except Continue:
                        continue
                    except Break:
                        break
            elif isinstance(statement, ast.SwitchStatement):
                target = value(statement.target, frame)
                selected = next((case for labels, case in statement.cases
                                 if target in [value(label, frame) for label in labels]), statement.default)
                if selected is not None:
                    block(selected.statements, Frame(frame))
            elif isinstance(statement, ast.ReturnStatement):
                raise Return(value(statement.expression, frame) if statement.expression is not None else None)
            elif isinstance(statement, ast.BreakStatement):
                raise Break
            elif isinstance(statement, ast.ContinueStatement):
                raise Continue
            elif isinstance(statement, ast.EndStatement):
                raise End
            elif not isinstance(statement, (ast.Include, ast.QubitDeclaration, ast.QuantumBarrier)):
                raise AssertionError(f'unexpected fixture statement: {statement}')
    try:
        block(program.statements, global_frame)
    except End:
        pass
    values = {name: v for name, v in global_frame.values.items() if not name.startswith('__qasmviz_')}
    return state.data, probability, values
