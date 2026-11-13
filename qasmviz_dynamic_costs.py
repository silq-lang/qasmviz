"""Conservative execution counts for compiled OpenQASM control flow.

Each exit kind carries independent lower/upper bounds. Classical state and
branch correlations are not inferred. Unknown iteration bounds have a null
upper bound; depth is not estimated. A separate bounded pass adds exact
expectations where it can prove measurement probabilities and classical flow.
"""

from dataclasses import dataclass, field

from openqasm3 import ast

from qasmviz_constants import ConstantEvaluationError, ConstantEvaluator
from qasmviz_loops import iterations, nodes, writes_name


@dataclass(frozen=True)
class Bound:
    lower: int = 0
    upper: int | None = 0

    def add(self, other):
        return Bound(self.lower + other.lower,
                     None if self.upper is None or other.upper is None else self.upper + other.upper)

    def join(self, other):
        return Bound(min(self.lower, other.lower),
                     None if self.upper is None or other.upper is None else max(self.upper, other.upper))

    def scale(self, count):
        if count == 0:
            return Bound()
        return Bound(self.lower * count, None if self.upper is None else self.upper * count)

    def json(self):
        return {'lower': self.lower, 'upper': self.upper}


def combine(left, right, operation):
    return {key: operation(left.get(key, Bound()), right.get(key, Bound()))
            for key in left.keys() | right.keys()}


def add(left, right):
    return combine(left, right, Bound.add)


def join(left, right):
    return combine(left, right, Bound.join)


def merge(flow, exit_kind, cost):
    flow[exit_kind] = join(flow[exit_kind], cost) if exit_kind in flow else cost


def total(flow):
    result = None
    for cost in flow.values():
        result = cost if result is None else join(result, cost)
    return result or {}


@dataclass
class Scope:
    constants: dict = field(default_factory=dict)
    names: set = field(default_factory=set)
    sizes: dict = field(default_factory=dict)

    def child(self):
        return Scope(dict(self.constants), set(self.names), dict(self.sizes))

    def evaluator(self):
        return ConstantEvaluator(self.constants, self.names)

    def mask(self, name):
        self.constants.pop(name, None)
        self.sizes.pop(name, None)
        self.names.add(name)


class Analyzer:
    def __init__(self):
        self.width = 0
        # Counts for large loops use interval multiplication instead of
        # enumerating an arbitrarily large number of classical executions.
        self.remaining_iterations = 1024

    @staticmethod
    def truth(expression, scope):
        try:
            return bool(scope.evaluator().evaluate(expression).value)
        except ConstantEvaluationError:
            return None

    def operand_size(self, node, scope):
        if isinstance(node, ast.Identifier):
            return scope.sizes.get(node.name, Bound(0, None))
        if isinstance(node, ast.Concatenation):
            return self.operand_size(node.lhs, scope).add(self.operand_size(node.rhs, scope))
        if isinstance(node, ast.IndexedIdentifier):
            base = self.operand_size(node.name, scope)
            index = node.indices[0]
        elif isinstance(node, ast.IndexExpression):
            base = self.operand_size(node.collection, scope)
            index = node.index
        else:
            return Bound(0, None)
        if isinstance(index, list):
            if len(index) != 1:
                return Bound(0, None)
            index = index[0]
        if isinstance(index, ast.DiscreteSet):
            return Bound(len(index.values), len(index.values))
        if not isinstance(index, ast.RangeDefinition):
            return Bound(1, 1)
        try:
            evaluator = scope.evaluator()
            step = evaluator.integer(index.step) if index.step is not None else 1
            if step == 0 or base.upper is None or base.lower != base.upper:
                return Bound(0, None)
            size = base.lower
            start = evaluator.integer(index.start) if index.start is not None else (0 if step > 0 else size - 1)
            end = evaluator.integer(index.end) if index.end is not None else (size - 1 if step > 0 else 0)
            start = start + size if start < 0 else start
            end = end + size if end < 0 else end
            count = 0 if (end - start) * step < 0 else abs(end - start) // abs(step) + 1
            return Bound(count, count)
        except ConstantEvaluationError:
            return Bound(0, None)

    def block(self, statements, scope):
        flow = {'normal': {}}
        for statement in statements:
            prefix = flow.pop('normal', None)
            if prefix is None:
                break
            for exit_kind, cost in self.statement(statement, scope).items():
                merge(flow, exit_kind, add(prefix, cost))
        return flow

    def statement(self, statement, scope):
        cost = {}
        if isinstance(statement, ast.QuantumGate):
            name = statement.name.name
            cost = {'gate-count': Bound(1, 1), 'gate:' + name: Bound(1, 1)}
            if name in ('t', 'tdg'):
                cost['t-count'] = Bound(1, 1)
            if name == 'cx':
                cost['cx-count'] = Bound(1, 1)
        elif isinstance(statement, ast.QuantumMeasurementStatement):
            cost['measurements'] = self.operand_size(statement.measure.qubit, scope)
        elif isinstance(statement, ast.QuantumReset):
            cost['resets'] = self.operand_size(statement.qubits, scope)
        elif isinstance(statement, ast.QubitDeclaration):
            size = scope.evaluator().integer(statement.size) if statement.size is not None else 1
            scope.sizes[statement.qubit.name] = Bound(size, size)
            self.width += size
        elif isinstance(statement, ast.ConstantDeclaration):
            evaluator = scope.evaluator()
            try:
                value = evaluator.cast(statement.type, evaluator.evaluate(statement.init_expression))
            except ConstantEvaluationError as exc:
                value = exc
            scope.mask(statement.identifier.name)
            scope.constants[statement.identifier.name] = value
        elif isinstance(statement, (ast.ClassicalDeclaration, ast.IODeclaration)):
            if isinstance(getattr(statement, 'init_expression', None), ast.QuantumMeasurement):
                cost['measurements'] = self.operand_size(statement.init_expression.qubit, scope)
            scope.mask(statement.identifier.name)
        elif isinstance(statement, ast.AliasStatement):
            size = self.operand_size(statement.value, scope)
            scope.mask(statement.target.name)
            scope.sizes[statement.target.name] = size
        elif isinstance(statement, (ast.SubroutineDefinition, ast.ExternDeclaration)):
            scope.mask(statement.name.name)
        elif isinstance(statement, ast.BranchingStatement):
            condition = self.truth(statement.condition, scope)
            flow = {}
            for possible, body in ((condition is not False, statement.if_block),
                                   (condition is not True, statement.else_block)):
                if possible:
                    for exit_kind, result in self.block(body, scope.child()).items():
                        merge(flow, exit_kind, result)
            return self.call_exit(flow, statement.condition)
        elif isinstance(statement, ast.CompoundStatement):
            return self.block(statement.statements, scope.child())
        elif isinstance(statement, ast.SwitchStatement):
            return self.switch(statement, scope)
        elif isinstance(statement, ast.ForInLoop):
            return self.for_loop(statement, scope)
        elif isinstance(statement, ast.WhileLoop):
            condition = self.truth(statement.while_condition, scope)
            if condition is False:
                return {'normal': {}}
            body = self.block(statement.block, scope.child())
            return self.call_exit(self.repeat(body, None, condition is True), statement.while_condition)
        elif isinstance(statement, ast.BreakStatement):
            return {'break': {}}
        elif isinstance(statement, ast.ContinueStatement):
            return {'continue': {}}
        elif isinstance(statement, (ast.EndStatement, ast.ReturnStatement)):
            return {'stop': {}}
        flow = {'normal': cost}
        if isinstance(statement, (ast.ClassicalAssignment, ast.ClassicalDeclaration,
                                  ast.ExpressionStatement, ast.QuantumMeasurementStatement,
                                  ast.QuantumReset, ast.QuantumBarrier, ast.DelayInstruction,
                                  ast.AliasStatement)):
            flow = self.call_exit(flow, statement)
        return flow

    @staticmethod
    def call_exit(flow, expression):
        # A classical call may terminate the program or fail to return.
        # Quantum subroutines have already been inlined by source compilation.
        if any(isinstance(node, ast.FunctionCall) for node in nodes(expression)):
            merge(flow, 'stop', {})
        return flow

    def switch(self, statement, scope):
        try:
            target = scope.evaluator().integer(statement.target)
            selected = None
            for labels, case in statement.cases:
                if target in [scope.evaluator().integer(label) for label in labels]:
                    selected = case
                    break
            selected = selected or statement.default
            flow = self.block(selected.statements, scope.child()) if selected is not None else {'normal': {}}
        except ConstantEvaluationError:
            flow = {}
            cases = [case for _, case in statement.cases] + [statement.default]
            for case in cases:
                for exit_kind, cost in (self.block(case.statements, scope.child()) if case else {'normal': {}}).items():
                    merge(flow, exit_kind, cost)
        return self.call_exit(flow, statement.target)

    def for_loop(self, statement, scope):
        try:
            space = iterations(statement, scope.constants, scope.names)
        except ConstantEvaluationError:
            space = None
        child = scope.child()
        name = statement.identifier.name
        child.mask(name)
        if space is None or space.count > self.remaining_iterations:
            body = self.block(statement.block, child)
            return self.call_exit(self.repeat(body, space.count if space is not None else None),
                                  statement.set_declaration)
        self.remaining_iterations -= space.count
        flow, pending = {}, {}
        mutable = writes_name(statement.block, name)
        for index in range(space.count):
            if pending is None:
                break
            iteration = child.child()
            if not mutable:
                iteration.constants[name] = space.at(index)
            next_pending = None
            for exit_kind, cost in self.block(statement.block, iteration).items():
                cost = add(pending, cost)
                if exit_kind in ('normal', 'continue'):
                    next_pending = cost if next_pending is None else join(next_pending, cost)
                else:
                    merge(flow, 'normal' if exit_kind == 'break' else exit_kind, cost)
            pending = next_pending
        if pending is not None:
            merge(flow, 'normal', pending)
        return self.call_exit(flow, statement.set_declaration)

    @staticmethod
    def repeat(body, count, always=False):
        if count == 0:
            return {'normal': {}}
        repeats = 'normal' in body or 'continue' in body
        if not repeats:
            flow = {}
            for exit_kind, cost in body.items():
                merge(flow, 'normal' if exit_kind == 'break' else exit_kind, cost)
            if count is None and not always:
                merge(flow, 'normal', {})
            return flow
        all_costs = total(body)
        if count is None:
            bounds = {key: Bound(0, 0 if value.upper == 0 else None) for key, value in all_costs.items()}
        else:
            bounds = {key: value.scale(count) for key, value in all_costs.items()}
            if 'break' in body or 'stop' in body:
                bounds = {key: Bound(0, value.upper) for key, value in bounds.items()}
        flow = {}
        if not always or 'break' in body:
            flow['normal'] = bounds
        if count is None or 'stop' in body:
            flow['stop'] = bounds
        return flow


def collect_dynamic_costs(program):
    analyzer = Analyzer()
    counts = total(analyzer.block(program.statements, Scope()))
    declarations = Analyzer()
    declarations.block([statement for statement in program.statements if isinstance(
        statement, (ast.QubitDeclaration, ast.ConstantDeclaration, ast.ClassicalDeclaration,
                    ast.IODeclaration, ast.SubroutineDefinition, ast.ExternDeclaration)
    )], Scope())
    result = {
        'analysis': 'dynamic', 'bounds': 'conservative', 'width': declarations.width,
        'depth': None, 't-depth': None,
    }
    for key in ('gate-count', 't-count', 'cx-count', 'measurements', 'resets'):
        result[key] = counts.get(key, Bound()).json()
    result['gate-counts'] = {key[5:]: value.json() for key, value in sorted(counts.items())
                           if key.startswith('gate:') and value.upper != 0}
    from qasmviz_expected import attach_expected_counts
    from qasmviz_expected_qasm import execute_qasm
    return attach_expected_counts(result, lambda budget: execute_qasm(program, budget))


def print_dynamic_costs(costs):
    metrics = ('gate-count', 't-count', 'cx-count', '2q-count', 'measurements', 'resets')
    print(f"width: {costs['width']}")
    print('Execution-count bounds (conservative):')
    for key in metrics:
        if key not in costs:
            continue
        low, high = costs[key]['lower'], costs[key]['upper']
        value = f'{low}..unknown' if high is None else (str(low) if low == high else f'{low}..{high}')
        print(f'  {key}: {value}')
    expected = costs.get('expected-counts')
    rows = []
    if expected is not None:
        for key in metrics:
            value = expected.get(key)
            if value is None:
                continue
            numerator, denominator = value['numerator'], value['denominator']
            bound = costs.get(key)
            if (bound is not None and bound['lower'] == bound['upper']
                    and numerator == bound['lower'] * denominator):
                continue
            whole, remainder = divmod(numerator, denominator)
            text = str(whole) if not remainder else f'{whole} + {remainder}/{denominator}'
            rows.append((key, text))
    if rows:
        print('Expected execution counts (qubits initially zero):')
        for key, text in rows:
            print(f'  {key}: {text}')
        if 'reason' in expected:
            print('Expected-count analysis stopped: ' + expected['reason'] + '.')
    print('Depth is not estimated.')
