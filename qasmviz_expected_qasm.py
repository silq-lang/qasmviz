"""Small classical interpreter for expectations of compiled OpenQASM 3.

Only initialized scalar arithmetic, bit arrays and bounded control flow are
executed. Unsupported classical semantics stop expectation analysis; they do
not affect compilation or the conservative execution-count bounds.
"""

from dataclasses import dataclass, fields

from openqasm3 import ast

from qasmviz_constants import ConstantEvaluationError, ConstantEvaluator, Scalar
from qasmviz_expected import Event, NotEasy, Path, gate, measure
from qasmviz_loops import iterations


@dataclass(frozen=True)
class Cell:
    type: object
    value: object = None


class Evaluator(ConstantEvaluator):
    def __init__(self, path, budget):
        super().__init__({}, path.values.keys())
        self.path, self.budget, self.level = path, budget, 0

    def evaluate(self, node):
        self.budget.spend()
        self.level += 1
        self.budget.depth(self.level)
        try:
            value = super().evaluate(node)
            if value.kind == 'float':
                raise NotEasy('floating-point classical computation')
            return value
        finally:
            self.level -= 1

    def _evaluate(self, node):
        if isinstance(node, ast.Identifier):
            cell = self.path.values.get(node.name)
            if cell is None or cell.value is None or cell.type == 'qubits':
                raise NotEasy(f'classical value is unknown: {node.name}')
            if isinstance(cell.value, tuple):
                if any(v is None for v in cell.value):
                    raise NotEasy(f'classical value is unknown: {node.name}')
                return Scalar(sum(int(v.value) << i for i, v in enumerate(cell.value)), 'uint', len(cell.value))
            return cell.value
        if isinstance(node, ast.IndexExpression):
            name, indices = selected(node, self.path, self.budget)
            values = self.path.values[name].value
            if len(indices) != 1 or values[indices[0]] is None:
                raise NotEasy('classical slice or uninitialized bit in expression')
            return values[indices[0]]
        if isinstance(node, ast.BinaryExpression) and node.op.name in ('&&', '||'):
            left = bool(self.evaluate(node.lhs).value)
            if (node.op.name == '&&' and not left) or (node.op.name == '||' and left):
                return Scalar(left, 'bool')
            return Scalar(bool(self.evaluate(node.rhs).value), 'bool')
        if isinstance(node, ast.FunctionCall):
            raise NotEasy('classical function call')
        return super()._evaluate(node)

    def cast(self, type_node, value, *, explicit=False):
        if isinstance(type_node, ast.BitType) and type_node.size is None:
            if value.value not in (0, 1):
                raise NotEasy('non-bit value assigned to a bit')
            return Scalar(bool(value.value), 'bool')
        if not isinstance(type_node, (ast.BoolType, ast.IntType, ast.UintType)):
            raise NotEasy('unsupported classical type')
        return super().cast(type_node, value, explicit=explicit)


def selected(node, path, budget):
    """Resolve a one-dimensional identifier/slice without allocating by width."""
    budget.spend()
    if isinstance(node, ast.Identifier):
        name, index = node.name, None
    elif isinstance(node, ast.IndexedIdentifier) and len(node.indices) == 1:
        name, index = node.name.name, node.indices[0]
    elif isinstance(node, ast.IndexExpression) and isinstance(node.collection, ast.Identifier):
        name, index = node.collection.name, node.index
    else:
        raise NotEasy('unsupported indexed operand')
    cell = path.values.get(name)
    if cell is None:
        raise NotEasy(f'unknown operand: {name}')
    size = len(cell.value) if isinstance(cell.value, tuple) else 1
    if isinstance(index, list):
        if len(index) != 1:
            raise NotEasy('multidimensional operand')
        index = index[0]
    evaluator = Evaluator(path, budget)
    if index is None:
        indices = range(size)
    elif isinstance(index, ast.DiscreteSet):
        if len(index.values) > budget.limits.qubits:
            raise NotEasy('operand size limit reached')
        indices = [evaluator.integer(v) for v in index.values]
    elif isinstance(index, ast.RangeDefinition):
        step = evaluator.integer(index.step) if index.step is not None else 1
        if not step:
            raise NotEasy('zero slice step')
        start = evaluator.integer(index.start) if index.start is not None else (0 if step > 0 else size - 1)
        end = evaluator.integer(index.end) if index.end is not None else (size - 1 if step > 0 else 0)
        indices = range(start, end + (1 if step > 0 else -1), step)
    else:
        indices = [evaluator.integer(index)]
    if len(indices) > budget.limits.qubits or any(not 0 <= i < size for i in indices):
        raise NotEasy('operand index out of range or size limit reached')
    return name, tuple(indices)


class Executor:
    def __init__(self, program, budget):
        self.program, self.budget = program, budget
        self.qubits = 0
        self.definitions = {}

    def preflight(self):
        # Reject expensive/unsupported constructs before any probability work.
        pending = [(self.program, 0)]
        while pending:
            node, depth = pending.pop()
            self.budget.spend()
            self.budget.depth(depth)
            if isinstance(node, (ast.WhileLoop, ast.ExternDeclaration, ast.SubroutineDefinition,
                                 ast.FunctionCall, ast.IODeclaration)):
                raise NotEasy(f'unsupported expectation construct: {type(node).__name__}')
            if isinstance(node, ast.ForInLoop):
                try:
                    count = iterations(node, {}).count
                except ConstantEvaluationError:
                    pass  # Runtime scalar values may still give a small loop.
                else:
                    if count > self.budget.limits.iterations:
                        raise NotEasy('loop iteration limit reached')
            if isinstance(node, ast.QASMNode):
                pending.extend((getattr(node, f.name), depth + 1) for f in fields(node) if f.name != 'span')
            elif isinstance(node, (list, tuple)):
                if len(node) > self.budget.limits.steps - self.budget.steps:
                    raise NotEasy('analysis work limit reached')
                pending.extend((item, depth) for item in node)

    def run(self):
        try:
            self.preflight()
            return self.block(self.program.statements, [Path()])
        except ConstantEvaluationError as exc:
            raise NotEasy(f'unsupported classical evaluation: {exc}') from exc

    def bind(self, path, name, cell):
        if path.scopes and name not in path.scopes[-1]:
            path.scopes[-1][name] = path.values.get(name)
        path.values[name] = cell

    def block(self, statements, paths, *, scoped=False, depth=0):
        self.budget.depth(depth)
        if scoped:
            for path in paths:
                path.scopes.append({})
        for statement in statements:
            next_paths = []
            for path in paths:
                self.budget.spend()
                next_paths.extend(self.statement(statement, path, depth) if path.exit == 'normal' else [path])
            paths = next_paths
            if all(path.exit != 'normal' for path in paths):
                break
        if scoped:
            for path in paths:
                for name, old in path.scopes.pop().items():
                    if old is None:
                        path.values.pop(name, None)
                    else:
                        path.values[name] = old
        return paths

    def wires(self, operand, path):
        if isinstance(operand, ast.Concatenation):
            return self.wires(operand.lhs, path) + self.wires(operand.rhs, path)
        name, indices = selected(operand, path, self.budget)
        cell = path.values[name]
        if cell.type != 'qubits':
            raise NotEasy('quantum operand is not a qubit')
        return tuple(cell.value[i] for i in indices)

    def assign(self, target, value, path):
        name, indices = selected(target, path, self.budget)
        cell = path.values[name]
        evaluator = Evaluator(path, self.budget)
        if isinstance(cell.value, tuple):
            values = list(cell.value)
            incoming = value if isinstance(value, tuple) else (value,)
            if len(incoming) != len(indices):
                raise NotEasy('classical assignment width mismatch')
            for i, v in zip(indices, incoming):
                values[i] = evaluator.cast(ast.BitType(), v)
            path.values[name] = Cell(cell.type, tuple(values))
        else:
            if isinstance(value, tuple):
                if len(value) != 1:
                    raise NotEasy('classical assignment width mismatch')
                value = value[0]
            path.values[name] = Cell(cell.type, evaluator.cast(cell.type, value))

    def measurement(self, operand, target, path):
        outcomes = [(path, ())]
        for wire in self.wires(operand, path):
            outcomes = [(child, values + (Scalar(bool(bit), 'bool'),))
                        for before, values in outcomes for child, bit in measure(before, wire, self.budget)]
        for child, values in outcomes:
            if target is not None:
                self.assign(target, values, child)
        return [child for child, _ in outcomes]

    def quantum_gate(self, statement, path, depth=0, *, count=True):
        self.budget.depth(depth)
        name = statement.name.name
        operands = [self.wires(q, path) for q in statement.qubits]
        length = max(map(len, operands), default=0)
        if not length or any(len(q) not in (1, length) for q in operands):
            raise NotEasy('unsupported gate broadcast')
        # The source bounds count each compiled gate statement once. Compiler
        # output normally already expands gate broadcasting.
        if count:
            path.count('gate-count')
            path.count('gate:' + name)
            path.count('t-count', int(name in ('t', 'tdg')))
            if len(operands) == 2:
                path.count('2q-count')
            if name == 'cx':
                path.count('cx-count')
        for index in range(length):
            wires = tuple(q[index if len(q) > 1 else 0] for q in operands)
            if statement.modifiers or statement.arguments:
                # Compiled Clifford+T output has neither. A parameterized
                # diagonal can still be skipped by a commutation proof.
                diagonal = name in ('p', 'u1', 'rz', 'cp', 'cu1') and not statement.modifiers
                gate(path, 'diagonal' if diagonal else 'unsupported:' + name,
                     wires, self.budget, None, count=False)
            elif name in self.definitions:
                definition = self.definitions[name]
                if definition.arguments or len(definition.qubits) != len(wires):
                    raise NotEasy('parameterized gate definition')
                path.scopes.append({})
                for formal, wire in zip(definition.qubits, wires):
                    self.bind(path, formal.name, Cell('qubits', (wire,)))
                for item in definition.body:
                    self.budget.spend()
                    if isinstance(item, ast.QuantumGate):
                        self.quantum_gate(item, path, depth + 1, count=False)
                    elif not isinstance(item, ast.QuantumPhase) or item.modifiers or item.qubits:
                        raise NotEasy('unsupported gate definition body')
                for formal, old in path.scopes.pop().items():
                    if old is None:
                        path.values.pop(formal, None)
                    else:
                        path.values[formal] = old
            else:
                gate(path, name, wires, self.budget, None, count=False)

    def statement(self, statement, path, depth):
        evaluator = Evaluator(path, self.budget)
        if isinstance(statement, ast.QubitDeclaration):
            size = evaluator.integer(statement.size) if statement.size is not None else 1
            if size < 1 or self.qubits + size > self.budget.limits.qubits:
                raise NotEasy('qubit limit reached')
            self.budget.spend(size)
            self.bind(path, statement.qubit.name, Cell('qubits', tuple(range(self.qubits, self.qubits + size))))
            self.qubits += size
        elif isinstance(statement, ast.QuantumGateDefinition):
            self.definitions[statement.name.name] = statement
        elif isinstance(statement, ast.QuantumGate):
            self.quantum_gate(statement, path)
        elif isinstance(statement, ast.QuantumPhase):
            if statement.modifiers or statement.qubits:
                raise NotEasy('controlled phase in source expectation analysis')
        elif isinstance(statement, ast.QuantumMeasurementStatement):
            return self.measurement(statement.measure.qubit, statement.target, path)
        elif isinstance(statement, ast.QuantumReset):
            for wire in self.wires(statement.qubits, path):
                path.count('resets')
                path.record(Event('reset', (wire,)), self.budget)
        elif isinstance(statement, (ast.ClassicalDeclaration, ast.ConstantDeclaration)):
            type_node = statement.type
            size = getattr(type_node, 'size', None)
            if size is not None:
                size = evaluator.integer(size)
                if not 1 <= size <= self.budget.limits.integer_bits:
                    raise NotEasy('classical size limit reached')
                type_node = type(type_node)(ast.IntegerLiteral(size))
            if isinstance(type_node, ast.BitType) and size:
                self.budget.spend(size)
                value = (None,) * size
            else:
                value = None
            self.bind(path, statement.identifier.name, Cell(type_node, value))
            init = statement.init_expression
            if isinstance(init, ast.QuantumMeasurement):
                return self.measurement(init.qubit, statement.identifier, path)
            if init is not None:
                self.assign(statement.identifier, evaluator.evaluate(init), path)
        elif isinstance(statement, ast.ClassicalAssignment):
            value = evaluator.evaluate(statement.rvalue)
            if statement.op.name != '=':
                if not isinstance(statement.lvalue, ast.Identifier):
                    raise NotEasy('indexed compound assignment')
                value = evaluator.binary(statement, statement.op.name[:-1],
                                         evaluator.evaluate(statement.lvalue), value)
            self.assign(statement.lvalue, value, path)
        elif isinstance(statement, ast.AliasStatement):
            self.bind(path, statement.target.name, Cell('qubits', self.wires(statement.value, path)))
        elif isinstance(statement, ast.BranchingStatement):
            body = statement.if_block if evaluator.evaluate(statement.condition).value else statement.else_block
            return self.block(body, [path], scoped=True, depth=depth + 1)
        elif isinstance(statement, ast.CompoundStatement):
            return self.block(statement.statements, [path], scoped=True, depth=depth + 1)
        elif isinstance(statement, ast.SwitchStatement):
            value = evaluator.integer(statement.target)
            body = statement.default
            for labels, case in statement.cases:
                if value in [evaluator.integer(label) for label in labels]:
                    body = case
                    break
            return self.block(body.statements, [path], scoped=True, depth=depth + 1) if body else [path]
        elif isinstance(statement, ast.ForInLoop):
            self.budget.spend(len(path.values))
            bindings = {name: cell.value for name, cell in path.values.items() if isinstance(cell.value, Scalar)}
            space = iterations(statement, bindings, set(path.values))
            if space.count > self.budget.limits.iterations:
                raise NotEasy('loop iteration limit reached')
            paths, finished = [path], []
            for index in range(space.count):
                for current in paths:
                    current.scopes.append({})
                    self.bind(current, statement.identifier.name, Cell(statement.type, space.at(index)))
                body = self.block(statement.block, paths, depth=depth + 1)
                paths = []
                for current in body:
                    for name, old in current.scopes.pop().items():
                        if old is None:
                            current.values.pop(name, None)
                        else:
                            current.values[name] = old
                    if current.exit == 'break':
                        current.exit = 'normal'
                        finished.append(current)
                    elif current.exit == 'stop':
                        finished.append(current)
                    else:
                        current.exit = 'normal'
                        paths.append(current)
                if not paths:
                    break
            return finished + paths
        elif isinstance(statement, (ast.BreakStatement, ast.ContinueStatement, ast.EndStatement)):
            path.exit = {ast.BreakStatement: 'break', ast.ContinueStatement: 'continue', ast.EndStatement: 'stop'}[type(statement)]
        elif not isinstance(statement, (ast.Include, ast.QuantumBarrier)):
            raise NotEasy(f'unsupported expectation statement: {type(statement).__name__}')
        return [path]


def execute_qasm(program, budget):
    return Executor(program, budget).run()
