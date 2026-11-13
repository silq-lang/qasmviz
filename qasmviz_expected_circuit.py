"""Bounded classical execution of Qiskit circuits for exact expected counts."""

import operator

from qiskit.circuit import (
    Barrier, BoxOp, BreakLoopOp, CASE_DEFAULT, ClassicalRegister, Clbit, ContinueLoopOp,
    ControlledGate, ControlFlowOp, Delay, ForLoopOp, Gate, IfElseOp, Measure, Reset, Store, SwitchCaseOp, WhileLoopOp,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import GlobalPhaseGate, get_standard_gate_name_mapping

from qasmviz_expected import Event, NotEasy, Path, canonical_phase, gate, measure
from qasmviz_expected_stream import Choice, Quantum, Sample, UnsupportedStream, execute, plan


_STANDARD = get_standard_gate_name_mapping()
_BINARY = {
    'BIT_AND': operator.and_, 'BIT_OR': operator.or_, 'BIT_XOR': operator.xor,
    'EQUAL': operator.eq, 'NOT_EQUAL': operator.ne, 'LESS': operator.lt,
    'LESS_EQUAL': operator.le, 'GREATER': operator.gt, 'GREATER_EQUAL': operator.ge,
}


class Executor:
    def __init__(self, circuit, t_cost, budget):
        self.circuit, self.t_cost, self.budget = circuit, t_cost, budget

    def preflight(self):
        pending = [(self.circuit, 0)]
        while pending:
            circuit, depth = pending.pop()
            self.budget.depth(depth)
            self.budget.input(len(circuit.data))
            if len(circuit.data) > self.budget.remaining:
                raise NotEasy('analysis work limit reached')
            for instruction in circuit.data:
                self.budget.spend()
                operation = instruction.operation
                if isinstance(operation, WhileLoopOp):
                    raise NotEasy('unsupported expectation construct: while loop')
                if isinstance(operation, ForLoopOp):
                    try:
                        count = len(operation.params[0])
                    except OverflowError as exc:
                        raise NotEasy('loop iteration limit reached') from exc
                    if count > self.budget.limits.iterations:
                        raise NotEasy('loop iteration limit reached')
                if isinstance(operation, ControlFlowOp):
                    if len(operation.blocks) > self.budget.remaining:
                        raise NotEasy('analysis work limit reached')
                    pending.extend((block, depth + 1) for block in operation.blocks)

    def run(self):
        self.preflight()
        self.budget.input(self.circuit.num_qubits + self.circuit.num_clbits)
        self.budget.spend(self.circuit.num_qubits + self.circuit.num_clbits)
        qmap = {q: i for i, q in enumerate(self.circuit.qubits)}
        cmap = {c: i for i, c in enumerate(self.circuit.clbits)}
        try:
            stream = plan(self.stream(qmap, cmap), self.budget)
        except UnsupportedStream:
            pass
        else:
            return execute(stream, self.budget)
        return self.block(self.circuit, [Path()], qmap, cmap)

    def condition_bit(self, condition, cmap, depth=0):
        """Accept a single bit, its negation, or equality to a Boolean."""
        self.budget.spend()
        self.budget.depth(depth)
        if isinstance(condition, Clbit):
            return cmap[condition], True
        if isinstance(condition, ClassicalRegister) and len(condition) == 1:
            return cmap[condition[0]], True
        if isinstance(condition, tuple) and condition[1] in (0, 1):
            key, polarity = self.condition_bit(condition[0], cmap, depth + 1)
            return key, polarity == bool(condition[1])
        if isinstance(condition, expr.Var):
            return self.condition_bit(condition.var, cmap, depth + 1)
        if isinstance(condition, expr.Unary) and condition.op.name == 'LOGIC_NOT':
            key, polarity = self.condition_bit(condition.operand, cmap, depth + 1)
            return key, not polarity
        if isinstance(condition, expr.Binary) and condition.op.name in ('EQUAL', 'NOT_EQUAL'):
            left, right = condition.left, condition.right
            if isinstance(left, expr.Value):
                left, right = right, left
            if isinstance(right, expr.Value) and right.value in (0, 1):
                key, polarity = self.condition_bit(left, cmap, depth + 1)
                return key, (polarity == bool(right.value)) == (condition.op.name == 'EQUAL')
        raise UnsupportedStream()

    def quantum(self, instructions, qmap, cmap):
        path = Path(track=False)
        for instruction in instructions:
            self.budget.spend()
            operation = instruction.operation
            if isinstance(operation, (ControlFlowOp, Store, Measure)):
                raise UnsupportedStream()
            self.instruction(instruction, path, qmap, cmap, 0)
        return Quantum.capture(path)

    def stream(self, qmap, cmap):
        items = []
        for instruction in self.circuit.data:
            self.budget.spend()
            operation = instruction.operation
            if operation.base_class is Measure:
                if len(instruction.qubits) != 1 or len(instruction.clbits) != 1:
                    raise UnsupportedStream()
                items.append(Sample(qmap[instruction.qubits[0]], cmap[instruction.clbits[0]]))
            elif isinstance(operation, IfElseOp):
                key, polarity = self.condition_bit(operation.condition, cmap)
                branches = []
                for block in operation.blocks:
                    branches.append(self.quantum(block.data,
                        {q: qmap[outer] for q, outer in zip(block.qubits, instruction.qubits)},
                        {c: cmap[outer] for c, outer in zip(block.clbits, instruction.clbits)}))
                if len(branches) == 1:
                    branches.append(Quantum((), {}))
                items.append(Choice(key, tuple(reversed(branches)) if polarity else tuple(branches)))
            else:
                item = self.quantum((instruction,), qmap, cmap)
                if item.events or item.counts:
                    items.append(item)
        return items

    def read(self, value, path, cmap, depth=0):
        self.budget.spend()
        self.budget.depth(depth)
        if isinstance(value, tuple):
            return self.read(value[0], path, cmap, depth + 1) == value[1]
        if isinstance(value, Clbit):
            key = cmap[value]
            if key not in path.values:
                raise NotEasy('classical input is unknown')
            return path.values[key]
        if isinstance(value, ClassicalRegister):
            return sum(int(self.read(bit, path, cmap, depth + 1)) << i for i, bit in enumerate(value))
        if isinstance(value, expr.Value):
            if not isinstance(value.value, (bool, int)):
                raise NotEasy('unsupported classical literal')
            if abs(value.value).bit_length() > self.budget.limits.integer_bits:
                raise NotEasy('classical size limit reached')
            return value.value
        if isinstance(value, expr.Var):
            if isinstance(value.var, (Clbit, ClassicalRegister)):
                return self.read(value.var, path, cmap, depth + 1)
            if value.var not in path.values:
                raise NotEasy('classical input is unknown')
            return path.values[value.var]
        if isinstance(value, expr.Index):
            target = self.read(value.target, path, cmap, depth + 1)
            index = self.read(value.index, path, cmap, depth + 1)
            if not 0 <= index < value.target.type.width:
                raise NotEasy('classical index out of range')
            return bool((target >> index) & 1)
        if isinstance(value, expr.Cast):
            raw = self.read(value.operand, path, cmap, depth + 1)
            if isinstance(value.type, types.Bool):
                return bool(raw)
            if isinstance(value.type, types.Uint) and value.type.width <= self.budget.limits.integer_bits:
                if 0 <= raw < (1 << value.type.width):
                    return int(raw)
            raise NotEasy('unsupported classical cast')
        if isinstance(value, expr.Unary):
            raw = self.read(value.operand, path, cmap, depth + 1)
            if value.op.name == 'LOGIC_NOT':
                return not raw
            if value.op.name == 'BIT_NOT' and isinstance(value.type, types.Bool):
                return not raw
            if value.op.name == 'BIT_NOT' and isinstance(value.type, types.Uint):
                if value.type.width > self.budget.limits.integer_bits:
                    raise NotEasy('classical size limit reached')
                return raw ^ ((1 << value.type.width) - 1)
        if isinstance(value, expr.Binary):
            left = self.read(value.left, path, cmap, depth + 1)
            name = value.op.name
            if name == 'LOGIC_AND' and not left:
                return False
            if name == 'LOGIC_OR' and left:
                return True
            right = self.read(value.right, path, cmap, depth + 1)
            if name in ('LOGIC_AND', 'LOGIC_OR'):
                return bool(right)
            if name in _BINARY:
                return _BINARY[name](left, right)
        raise NotEasy('unsupported classical expression')

    def write(self, target, value, path, cmap):
        if isinstance(target, expr.Var):
            target = target.var
        if isinstance(target, Clbit):
            path.values[cmap[target]] = bool(value)
        elif isinstance(target, ClassicalRegister):
            for i, bit in enumerate(target):
                path.values[cmap[bit]] = bool((value >> i) & 1)
        elif not isinstance(target, expr.Expr):
            path.values[target] = value
        else:
            raise NotEasy('unsupported classical assignment target')

    def block(self, circuit, paths, qmap, cmap, depth=0):
        self.budget.depth(depth)
        for instruction in circuit.data:
            next_paths = []
            for path in paths:
                self.budget.spend()
                if path.exit == 'normal':
                    next_paths.extend(self.instruction(instruction, path, qmap, cmap, depth))
                else:
                    next_paths.append(path)
            paths = next_paths
            if all(path.exit != 'normal' for path in paths):
                break
        return paths

    def subblock(self, block, instruction, paths, qmap, cmap, depth):
        return self.block(block, paths,
                          {q: qmap[outer] for q, outer in zip(block.qubits, instruction.qubits)},
                          {c: cmap[outer] for c, outer in zip(block.clbits, instruction.clbits)}, depth + 1)

    def instruction(self, instruction, path, qmap, cmap, depth):
        operation = instruction.operation
        if isinstance(operation, IfElseOp):
            index = 0 if self.read(operation.condition, path, cmap) else 1
            if index == len(operation.blocks):
                return [path]
            return self.subblock(operation.blocks[index], instruction, [path], qmap, cmap, depth)
        if isinstance(operation, SwitchCaseOp):
            value, default = self.read(operation.target, path, cmap), None
            for labels, block in operation.cases_specifier():
                if value in labels:
                    return self.subblock(block, instruction, [path], qmap, cmap, depth)
                if CASE_DEFAULT in labels:
                    default = block
            return self.subblock(default, instruction, [path], qmap, cmap, depth) if default is not None else [path]
        if isinstance(operation, ForLoopOp):
            paths, finished = [path], []
            for _ in operation.params[0]:
                body = self.subblock(operation.blocks[0], instruction, paths, qmap, cmap, depth)
                paths = []
                for current in body:
                    if current.exit == 'break':
                        current.exit = 'normal'
                        finished.append(current)
                    else:
                        current.exit = 'normal'
                        paths.append(current)
                if not paths:
                    break
            return finished + paths
        if isinstance(operation, BoxOp):
            return self.subblock(operation.blocks[0], instruction, [path], qmap, cmap, depth)
        if isinstance(operation, (BreakLoopOp, ContinueLoopOp)):
            path.exit = 'break' if isinstance(operation, BreakLoopOp) else 'continue'
            return [path]
        if isinstance(operation, ControlFlowOp):
            raise NotEasy(f'unsupported expectation control flow: {operation.name}')
        if isinstance(operation, Store):
            self.write(operation.lvalue, self.read(operation.rvalue, path, cmap), path, cmap)
            return [path]
        wires = tuple(qmap[q] for q in instruction.qubits)
        if operation.name == 'measure':
            if operation.base_class is not Measure or len(wires) != 1 or len(instruction.clbits) != 1:
                raise NotEasy('unsupported measurement instruction')
            outcomes = measure(path, wires[0], self.budget)
            for branch, value in outcomes:
                branch.values[cmap[instruction.clbits[0]]] = bool(value)
            return [branch for branch, _ in outcomes]
        if operation.name == 'reset':
            if operation.base_class is not Reset:
                raise NotEasy('unsupported reset instruction')
            for wire in wires:
                path.count('resets')
                path.record(Event('reset', (wire,)), self.budget)
            return [path]
        if operation.base_class in (Barrier, Delay, GlobalPhaseGate):
            return [path]
        if instruction.clbits or not wires:
            raise NotEasy('unsupported instruction with classical effects')
        if not isinstance(operation, Gate):
            raise NotEasy('unsupported quantum instruction')
        canonical = operation.name.split('_o')[0] if isinstance(operation, ControlledGate) else operation.name
        standard = _STANDARD.get(canonical)
        name = canonical if standard is not None and operation.base_class is standard.base_class else 'unsupported:' + operation.name
        path.count('gate-count')
        path.count('gate:' + operation.name)
        normalized = (canonical_phase(name, wires, operation.params[0])
                      if len(operation.params) == 1 else None)
        path.count('t-count', normalized[1] if normalized is not None else self.t_cost(operation))
        if len(wires) == 2:
            path.count('2q-count')
        if operation.name == 'cx':
            path.count('cx-count')
        open_controls = []
        if not name.startswith('unsupported:') and isinstance(operation, ControlledGate):
            open_controls = [wires[i] for i in range(operation.num_ctrl_qubits) if not (operation.ctrl_state >> i) & 1]
        for wire in open_controls:
            gate(path, 'x', (wire,), self.budget, None, count=False)
        if normalized is None:
            gate(path, name, wires, self.budget, None, count=False)
        else:
            for event in normalized[0]:
                path.record(event, self.budget)
        for wire in reversed(open_controls):
            gate(path, 'x', (wire,), self.budget, None, count=False)
        return [path]


def execute_circuit(circuit, t_cost, budget):
    return Executor(circuit, t_cost, budget).run()
