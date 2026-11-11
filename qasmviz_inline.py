"""Bounded, hygienic inlining of quantum OpenQASM subroutines.

Classical arguments are copied, quantum arguments are references, and each
expansion has its own return boundary. This is not a general type checker:
recursive calls, array-reference parameters and general embedded expressions
are rejected rather than being reordered or assigned guessed semantics.
"""

import copy
import math
from bisect import bisect_left
from dataclasses import dataclass, field, fields

import openqasm3
from openqasm3 import ast

from qasmviz_constants import ConstantEvaluationError, ConstantEvaluator
from qasmviz_loops import nodes, writes_name


DEFAULT_MAX_INLINE_STATEMENTS = 10000


class InliningError(ValueError):
    pass


def error(node, message):
    span = getattr(node, 'span', None)
    prefix = f'{span.start_line}:{span.start_column + 1}: ' if span else ''
    return InliningError(prefix + message)


def identifier(name):
    return ast.Identifier(name)


def assign(target, value):
    if isinstance(value, ast.QuantumMeasurement):
        return ast.QuantumMeasurementStatement(value, target)
    return ast.ClassicalAssignment(target, ast.AssignmentOperator['='], value)


def as_expression(node):
    if isinstance(node, ast.IndexedIdentifier):
        result = node.name
        for index in node.indices:
            result = ast.IndexExpression(result, index)
        return result
    return node


def as_operand(node):
    if isinstance(node, ast.IndexExpression):
        base = as_operand(node.collection)
        if isinstance(base, ast.Identifier):
            return ast.IndexedIdentifier(base, [node.index])
        if isinstance(base, ast.IndexedIdentifier):
            return ast.IndexedIdentifier(base.name, base.indices + [node.index])
    return node


@dataclass
class QubitReference:
    expression: ast.Expression
    scalar: bool
    members: tuple | None
    bases: frozenset


@dataclass
class Scope:
    renames: dict = field(default_factory=dict)
    quantum_expressions: dict = field(default_factory=dict)
    qubits: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    names: set = field(default_factory=set)
    functions: dict = field(default_factory=dict)
    captures: dict = field(default_factory=dict)
    stable: set = field(default_factory=set)
    local: bool = False

    def child(self):
        return Scope(dict(self.renames), dict(self.quantum_expressions), dict(self.qubits),
                     dict(self.constants), set(self.names), dict(self.functions),
                     dict(self.captures), set(self.stable), self.local)

    def mask(self, name, replacement):
        self.renames[name] = replacement
        self.names.add(name)
        self.constants.pop(name, None)
        self.qubits.pop(name, None)
        self.quantum_expressions.pop(name, None)
        self.functions.pop(name, None)
        self.captures.pop(name, None)
        self.stable.discard(name)

    def evaluator(self):
        return ConstantEvaluator(self.constants, self.names)

    def stable_expression(self, expression):
        """Whether a scalar expression can safely replace a by-value copy.

        Merely lacking function calls is insufficient: an array element can
        change while a later argument is being evaluated. Restrict reuse to
        constants, immutable loop iterators and already checked substitutions.
        """
        if isinstance(expression, ast.Identifier):
            if expression.name in self.constants or expression.name in self.stable or expression.name in self.quantum_expressions:
                return True
            try:
                self.evaluator().evaluate(expression)  # Includes unshadowed pi/tau/euler.
                return True
            except ConstantEvaluationError:
                return False
        if isinstance(expression, (ast.BooleanLiteral, ast.IntegerLiteral, ast.FloatLiteral)):
            return True
        if isinstance(expression, ast.UnaryExpression):
            return self.stable_expression(expression.expression)
        if isinstance(expression, ast.BinaryExpression):
            return self.stable_expression(expression.lhs) and self.stable_expression(expression.rhs)
        if isinstance(expression, ast.Cast):
            return self.stable_expression(expression.argument)
        return False


@dataclass
class Definition:
    node: ast.SubroutineDefinition
    scope: Scope


def quantum_definitions(program):
    definitions = {item.name.name: item for item in program.statements
                   if isinstance(item, ast.SubroutineDefinition)}
    quantum = {name for name, item in definitions.items()
               if any(isinstance(node, (ast.QuantumStatement, ast.QuantumMeasurement,
                                        ast.QuantumArgument)) for node in nodes(item))}
    while True:
        more = {name for name, item in definitions.items()
                if any(isinstance(node, ast.FunctionCall) and node.name.name in quantum
                       for node in nodes(item))}
        if more <= quantum:
            return quantum
        quantum |= more


def always_returns(statements):
    for statement in statements:
        if isinstance(statement, (ast.ReturnStatement, ast.EndStatement)):
            return True
        if isinstance(statement, ast.BranchingStatement):
            if always_returns(statement.if_block) and always_returns(statement.else_block):
                return True
        if isinstance(statement, ast.CompoundStatement) and always_returns(statement.statements):
            return True
        if isinstance(statement, ast.SwitchStatement) and statement.default is not None:
            if all(always_returns(case.statements) for _, case in statement.cases) and always_returns(statement.default.statements):
                return True
    return False


class Inliner:
    def __init__(self, source, program, editor, limit, width):
        self.source = source
        self.program = program
        self.editor = editor
        self.limit = limit
        self.count = 0
        self.width = width
        self.qubit_count = 0
        self.serial = 0
        self.quantum = quantum_definitions(program)
        self.captured_names = {node.name for item in program.statements
                               if isinstance(item, ast.SubroutineDefinition) and item.name.name in self.quantum
                               for node in nodes(item) if isinstance(node, ast.Identifier)}
        self.global_variables = {node.identifier.name for node in program.statements
                                 if isinstance(node, (ast.ClassicalDeclaration, ast.IODeclaration))}
        self.global_variables |= {node.qubit.name for node in program.statements
                                  if isinstance(node, ast.QubitDeclaration)}
        self.global_variables |= {node.target.name for node in program.statements
                                  if isinstance(node, ast.AliasStatement)}
        self.used = {node.name for node in nodes(program) if isinstance(node, ast.Identifier)}
        # The reference AST rounds these literals to Python float. Retain their
        # complete tokens when printing newly expanded classical statements.
        literal_tokens = {ast.FloatLiteral: 'FloatLiteral', ast.ImaginaryLiteral: 'ImaginaryLiteral',
                          ast.DurationLiteral: 'TimingLiteral'}
        for node in nodes(program):
            if type(node) in literal_tokens and node.span is not None:
                start, end = editor.bounds(node, node)
                tokens = [t for t in editor.tokens[bisect_left(editor.starts, start):bisect_left(editor.starts, end)]
                          if editor.lexer_symbol(t.type) == literal_tokens[type(node)]]
                if len(tokens) == 1:
                    node._inline_literal = tokens[0].text

    def fresh(self, hint):
        while True:
            self.serial += 1
            name = f'__qasmviz_{hint}_{self.serial}'
            if name not in self.used and name not in self.source:
                self.used.add(name)
                return identifier(name)

    def print_statements(self, statements):
        replacements = {}

        def replace(value):
            if isinstance(value, ast.QASMNode) and hasattr(value, '_inline_literal'):
                marker = self.fresh('literal')
                replacements[marker.name] = value._inline_literal
                return marker
            if isinstance(value, ast.QASMNode):
                for f in fields(value):
                    if f.name not in ('span', 'annotations'):
                        setattr(value, f.name, replace(getattr(value, f.name)))
            elif isinstance(value, list):
                return [replace(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(replace(v) for v in value)
            return value

        output = openqasm3.dumps(ast.Program(replace(copy.deepcopy(statements)))).rstrip()
        # Replace complete identifier tokens, never marker prefixes.
        import re
        return re.sub(r'\b__qasmviz_literal_\d+\b', lambda m: replacements.get(m[0], m[0]), output)

    def rewrite(self, value, scope, *, quantum=False):
        if isinstance(value, ast.Identifier):
            if quantum and value.name in scope.quantum_expressions:
                return copy.deepcopy(scope.quantum_expressions[value.name])
            if value.name in scope.renames:
                return copy.deepcopy(scope.renames[value.name])
            if scope.local and value.name in self.global_variables:
                raise error(value, f'subroutine cannot capture mutable global or qubit {value.name!r}')
            if scope.local and value.name not in scope.functions:
                raise error(value, f'unbound name {value.name!r} in quantum subroutine')
            return copy.deepcopy(value)
        if isinstance(value, ast.IndexedIdentifier):
            base = self.rewrite(value.name, scope, quantum=quantum)
            result = as_expression(base)
            for index in value.indices:
                result = ast.IndexExpression(result, self.rewrite(index, scope, quantum=quantum))
            return as_operand(result)
        if isinstance(value, ast.QASMNode):
            result = copy.copy(value)
            callable_node = isinstance(value, (ast.FunctionCall, ast.QuantumGate))
            if callable_node and scope.local and value.name.name in scope.renames and value.name.name not in scope.functions:
                raise error(value, f'subroutine binding shadows callable {value.name.name!r}')
            for f in fields(value):
                if f.name not in ('span', 'annotations'):
                    if callable_node and f.name == 'name':
                        result.name = copy.deepcopy(value.name)
                    else:
                        setattr(result, f.name, self.rewrite(getattr(value, f.name), scope, quantum=quantum))
            return result
        if isinstance(value, list):
            return [self.rewrite(v, scope, quantum=quantum) for v in value]
        if isinstance(value, tuple):
            return tuple(self.rewrite(v, scope, quantum=quantum) for v in value)
        return value

    def reference(self, expression, scope):
        expression = as_expression(expression)
        if isinstance(expression, ast.Identifier) and expression.name in scope.qubits:
            return scope.qubits[expression.name]
        if isinstance(expression, ast.Concatenation):
            left, right = self.reference(expression.lhs, scope), self.reference(expression.rhs, scope)
            if left.members is None or right.members is None:
                raise error(expression, 'quantum array arguments require constant selections')
            return QubitReference(ast.Concatenation(left.expression, right.expression), False,
                                  left.members + right.members, left.bases | right.bases)
        if isinstance(expression, ast.IndexExpression):
            base = self.reference(expression.collection, scope)
            if base.scalar or base.members is None:
                raise error(expression, 'cannot index a scalar or unresolved quantum argument')
            index = expression.index
            if isinstance(index, list):
                if len(index) != 1:
                    raise error(expression, 'quantum arguments require one-dimensional indexing')
                index = index[0]
            evaluator = scope.evaluator()
            scalar = not isinstance(index, (ast.RangeDefinition, ast.DiscreteSet))
            try:
                if isinstance(index, ast.RangeDefinition):
                    step = evaluator.integer(index.step) if index.step is not None else 1
                    if step == 0:
                        raise error(index, 'quantum slice step must not be zero')
                    start = evaluator.integer(index.start) if index.start is not None else (0 if step > 0 else len(base.members) - 1)
                    end = evaluator.integer(index.end) if index.end is not None else (len(base.members) - 1 if step > 0 else 0)
                    indices = range(start, end + (1 if step > 0 else -1), step)
                elif isinstance(index, ast.DiscreteSet):
                    indices = [evaluator.integer(v) for v in index.values]
                else:
                    indices = [evaluator.integer(index)]
                if any(i < 0 or i >= len(base.members) for i in indices):
                    raise error(expression, 'quantum argument index is out of range')
                members = tuple(base.members[i] for i in indices)
            except ConstantEvaluationError:
                if not scalar:
                    raise error(expression, 'quantum array arguments require constant selections')
                members = None
            if any(isinstance(n, (ast.FunctionCall, ast.QuantumMeasurement)) for n in nodes(index)):
                raise error(index, 'quantum argument indices must be side-effect-free')
            return QubitReference(ast.IndexExpression(base.expression, self.rewrite(expression.index, scope, quantum=True)),
                                  scalar, members, base.bases)
        raise error(expression, 'expected a declared qubit, register, or quantum alias argument')

    def check_calls(self, value):
        for node in nodes(value):
            if isinstance(node, ast.FunctionCall) and node.name.name in self.quantum:
                raise error(node, 'quantum subroutine calls must be a whole statement, assignment, '
                            'initializer, return or control-flow test (possibly cast or negated); '
                            'split this expression into separate statements')

    def test_expression(self, expression, scope, stack):
        """Lower a direct control-flow call without moving other evaluations.

        Unary operators and casts evaluate only one operand. General binary
        expressions and argument lists need further lowering to preserve their
        left-to-right evaluation order and are deliberately left unsupported.
        """
        if isinstance(expression, ast.FunctionCall) and expression.name.name in self.quantum:
            prefix, result = self.expand(expression, scope, stack)
            if result is None:
                raise error(expression, 'void quantum subroutine cannot be used as a control-flow test')
            return prefix, result
        if isinstance(expression, (ast.UnaryExpression, ast.Cast)):
            field = 'expression' if isinstance(expression, ast.UnaryExpression) else 'argument'
            prefix, operand = self.test_expression(getattr(expression, field), scope, stack)
            result = copy.copy(expression)
            setattr(result, field, operand)
            if isinstance(expression, ast.Cast):
                self.check_calls(expression.type)
                result.type = self.rewrite(expression.type, scope)
            return prefix, result
        self.check_calls(expression)
        return [], self.rewrite(expression, scope)

    def replace_control(self, statement, replacement, starting_count, *, edit, scope_body):
        # Nested expansions are part of replacement: charge them only once.
        self.count = starting_count + sum(isinstance(n, ast.Statement) for n in nodes(replacement))
        self.check_budget(statement)
        if edit:
            self.editor.replace([statement], [replacement], scope_body=scope_body, printer=self.print_statements)

    def expand(self, call, caller, stack):
        starting_count = self.count
        name = call.name.name
        if name in stack:
            raise error(call, 'recursive quantum subroutines cannot be inlined: ' + ' -> '.join((*stack, name)))
        if len(stack) >= 64:
            raise error(call, 'quantum subroutine nesting exceeds 64 calls')
        info = caller.functions.get(name)
        if not isinstance(info, Definition):
            raise error(call, f'quantum subroutine {name!r} is shadowed or used before its definition')
        definition, scope = info.node, info.scope.child()
        scope.local = True
        if len(call.arguments) != len(definition.arguments):
            raise error(call, f'{name!r} expects {len(definition.arguments)} arguments, got {len(call.arguments)}')
        if definition.return_type is not None and not always_returns(definition.body):
            raise error(definition, 'non-void quantum subroutine must have an explicit return or end on every path')
        self.check_calls(call.arguments)
        # A renamed local cannot shadow a captured callable. A caller binding
        # with its original spelling can; reject instead of capturing it.
        captured_calls = {n.name.name for n in nodes(definition.body)
                          if isinstance(n, (ast.FunctionCall, ast.QuantumGate))}
        for callable_name in captured_calls:
            binding = scope.functions.get(callable_name)
            if (binding is None or caller.functions.get(callable_name) is not binding) and callable_name in caller.names:
                actual = caller.renames.get(callable_name, identifier(callable_name))
                if isinstance(actual, ast.Identifier) and actual.name == callable_name:
                    raise error(call, f'caller shadows captured callable {callable_name!r}; use distinct names')
        prefix, references = [], []
        for position, (formal, actual) in enumerate(zip(definition.arguments, call.arguments)):
            formal_name = formal.name.name
            if isinstance(formal, ast.QuantumArgument):
                ref = self.reference(actual, caller)
                if ref.members is None and any(isinstance(n, ast.FunctionCall) for n in nodes(call.arguments[position + 1:])):
                    # Scalar operands are substituted into the body. We cannot
                    # defer a runtime selection across a later argument that
                    # may mutate its inputs. Capturing arbitrary runtime index
                    # types requires type information this pass does not have.
                    raise error(actual, 'dynamic qubit selection may change while evaluating a later argument; '
                                'use a fixed qubit selection or split the computation explicitly')
                if formal.size is None:
                    if not ref.scalar:
                        raise error(actual, f'quantum argument {formal_name!r} requires a scalar qubit')
                    replacement = as_operand(ref.expression)
                else:
                    try:
                        size = scope.evaluator().integer(formal.size)
                    except ConstantEvaluationError as exc:
                        raise error(formal, f'quantum parameter size must be constant: {exc}') from exc
                    if ref.scalar or ref.members is None or size < 1 or len(ref.members) != size:
                        raise error(actual, f'quantum argument {formal_name!r} requires a register of size {size}')
                    replacement = self.fresh(formal_name)
                    prefix.append(ast.AliasStatement(replacement, copy.deepcopy(ref.expression)))
                if ref.members is not None and len(set(ref.members)) != len(ref.members):
                    raise error(actual, 'quantum argument aliases the same qubit more than once')
                for previous in references:
                    if previous.members is not None and ref.members is not None:
                        if set(previous.members) & set(ref.members):
                            raise error(actual, 'quantum arguments overlap; each qubit may be passed only once')
                    elif previous.bases & ref.bases:
                        raise error(actual, 'cannot prove quantum arguments are disjoint; use constant indices')
                references.append(ref)
                scope.mask(formal_name, replacement)
                scope.qubits[formal_name] = QubitReference(as_expression(replacement), ref.scalar, ref.members, ref.bases)
            else:
                if isinstance(formal.type, ast.ArrayReferenceType) or formal.access is not None:
                    raise error(formal, 'array-reference subroutine parameters are not yet supported')
                kind = self.rewrite(formal.type, scope)
                value = self.rewrite(actual, caller)
                replacement = self.fresh(formal_name)
                immutable = not writes_name(definition.body, formal_name)
                constant = None
                if immutable:
                    try:
                        constant = caller.evaluator().evaluate(actual)
                        constant = scope.evaluator().cast(formal.type, constant)
                    except ConstantEvaluationError:
                        constant = None
                scope.mask(formal_name, replacement)
                if constant is not None:
                    prefix.append(ast.ConstantDeclaration(kind, replacement, value))
                    scope.constants[formal_name] = constant
                else:
                    prefix.append(ast.ClassicalDeclaration(kind, replacement, value))
                    if immutable and caller.stable_expression(actual):
                        scope.quantum_expressions[formal_name] = ast.Cast(kind, self.rewrite(actual, caller, quantum=True))
        result = self.fresh('result') if definition.return_type is not None else None
        if result is not None:
            prefix.append(ast.ClassicalDeclaration(self.rewrite(definition.return_type, scope), result))
        returned = self.fresh('returned')
        prefix.append(ast.ClassicalDeclaration(ast.BoolType(), returned, ast.BooleanLiteral(False)))
        body = self.block(definition.body, scope, stack=(*stack, name), returns=(result, returned, definition.return_type))
        # The one-iteration loop supplies a function-local break target. Checks
        # after original loops propagate a return through nested loop scopes.
        boundary = ast.ForInLoop(ast.IntType(), self.fresh('once'),
                                 ast.RangeDefinition(ast.IntegerLiteral(0), ast.IntegerLiteral(0), None), body)
        prefix.append(boundary)
        # Child expansions are already inside prefix. Charge each generated
        # statement once, including nested calls, rather than once per ancestor.
        self.count = starting_count + sum(isinstance(n, ast.Statement) for n in nodes(prefix))
        self.check_budget(call)
        return prefix, result

    def check_budget(self, node):
        if self.count > self.limit:
            raise error(node, f'subroutine expansion exceeds --max-inline-statements {self.limit}; '
                        'use smaller helpers or raise the limit')

    def call_statement(self, statement, scope, stack, returns):
        expression = None
        if isinstance(statement, ast.ExpressionStatement):
            expression = statement.expression
        elif isinstance(statement, ast.ClassicalAssignment):
            expression = statement.rvalue
        elif isinstance(statement, ast.ClassicalDeclaration):
            expression = statement.init_expression
        elif isinstance(statement, ast.ReturnStatement):
            expression = statement.expression
        if not isinstance(expression, ast.FunctionCall) or expression.name.name not in self.quantum:
            return None
        prefix, result = self.expand(expression, scope, stack)
        charged = sum(isinstance(n, ast.Statement) for n in nodes(prefix))
        if result is None and not isinstance(statement, ast.ExpressionStatement):
            raise error(expression, 'void quantum subroutine cannot be used as a value')
        before = []
        if isinstance(statement, ast.ClassicalDeclaration):
            declaration = copy.copy(statement)
            declaration.init_expression = None
            before = self.block([declaration], scope, stack=stack, returns=returns)
            prefix.append(assign(self.rewrite(statement.identifier, scope), result))
        elif isinstance(statement, ast.ClassicalAssignment):
            if statement.op.name != '=' or any(isinstance(n, ast.FunctionCall) for n in nodes(statement.lvalue)):
                raise error(statement, 'quantum call assignments require = and a side-effect-free target')
            prefix.append(ast.ClassicalAssignment(self.rewrite(statement.lvalue, scope), statement.op, result))
        elif isinstance(statement, ast.ReturnStatement):
            if returns is None:
                raise error(statement, 'return outside an inlined quantum subroutine')
            destination, returned, _ = returns
            if destination is None:
                raise error(statement, 'void quantum subroutine cannot return a value')
            prefix.append(assign(destination, result))
            # Break must be outside the callee's boundary and inside the
            # caller's boundary; a compound block does not change its target.
            prefix.extend([assign(returned, ast.BooleanLiteral(True)), ast.BreakStatement()])
        replacement = before + [ast.CompoundStatement(prefix)]
        self.count += sum(isinstance(n, ast.Statement) for n in nodes(replacement)) - charged
        self.check_budget(statement)
        return replacement

    def block(self, statements, scope, *, stack=(), returns=None, edit=False, global_scope=False, scope_body=False):
        result = []
        for statement in statements:
            starting_count = self.count
            if getattr(statement, 'annotations', None):
                raise error(statement, 'annotations are not supported by source Clifford+T compilation')
            if isinstance(statement, ast.SubroutineDefinition):
                if not global_scope:
                    raise error(statement, 'subroutine definitions must be at global scope')
                if statement.name.name in self.quantum:
                    captured = scope.child()
                    captured.renames.update(captured.captures)
                    # Mutable globals are not visible in subroutine scope.
                    for name in self.global_variables - captured.constants.keys():
                        captured.renames.pop(name, None)
                        captured.qubits.pop(name, None)
                    for name, value in (('pi', math.pi), ('π', math.pi), ('tau', math.tau),
                                        ('τ', math.tau), ('euler', math.e), ('ℇ', math.e)):
                        if name not in captured.names:
                            captured.renames[name] = ast.FloatLiteral(value)
                    info = Definition(statement, captured)
                    captured.functions[statement.name.name] = info
                    scope.functions[statement.name.name] = info
                    scope.names.add(statement.name.name)
                    if edit:
                        self.editor.replace([statement], [])
                    continue
                self.check_calls(statement)
                scope.functions[statement.name.name] = statement
                scope.names.add(statement.name.name)
                result.append(statement)
                continue
            replacement = self.call_statement(statement, scope, stack, returns)
            if replacement is not None:
                if edit:
                    self.editor.replace([statement], replacement, scope_body=scope_body and len(statements) == 1,
                                        printer=self.print_statements)
                result.extend(replacement)
                continue
            # Containers are checked one expression/body at a time so a call
            # in an ordinary nested statement can be expanded at its own span.
            if isinstance(statement, ast.BranchingStatement):
                new = copy.copy(statement)
                prefix, new.condition = self.test_expression(statement.condition, scope, stack)
                new.if_block = self.block(statement.if_block, scope.child(), stack=stack, returns=returns, edit=edit and not prefix, scope_body=True)
                new.else_block = self.block(statement.else_block, scope.child(), stack=stack, returns=returns, edit=edit and not prefix, scope_body=True)
                if prefix:
                    new = ast.CompoundStatement(prefix + [new])
                    self.replace_control(statement, new, starting_count, edit=edit, scope_body=scope_body and len(statements) == 1)
            elif isinstance(statement, (ast.ForInLoop, ast.WhileLoop)):
                child = scope.child()
                new = copy.copy(statement)
                prefix = []
                if isinstance(statement, ast.ForInLoop):
                    self.check_calls(statement.set_declaration)
                    new.type = self.rewrite(statement.type, scope)
                    new.set_declaration = self.rewrite(statement.set_declaration, scope, quantum=True)
                    new.identifier = self.fresh(statement.identifier.name) if scope.local else statement.identifier
                    child.mask(statement.identifier.name, new.identifier)
                    if not writes_name(statement.block, statement.identifier.name):
                        child.stable.add(statement.identifier.name)
                else:
                    prefix, new.while_condition = self.test_expression(statement.while_condition, scope, stack)
                new.block = self.block(statement.block, child, stack=stack, returns=returns, edit=edit and not prefix, scope_body=True)
                if prefix:
                    # The call is at the loop head, so continue evaluates it
                    # again, while break and return do not make an extra call.
                    stop = ast.BranchingStatement(ast.UnaryExpression(ast.UnaryOperator['!'], new.while_condition),
                                                  [ast.BreakStatement()], [])
                    new.block = prefix + [stop, ast.CompoundStatement(new.block)]
                    new.while_condition = ast.BooleanLiteral(True)
                    self.replace_control(statement, new, starting_count, edit=edit, scope_body=scope_body and len(statements) == 1)
                result.append(new)
                if returns is not None:
                    result.append(ast.BranchingStatement(returns[1], [ast.BreakStatement()], []))
                continue
            elif isinstance(statement, ast.CompoundStatement):
                new = copy.copy(statement)
                new.statements = self.block(statement.statements, scope.child(), stack=stack, returns=returns, edit=edit)
            elif isinstance(statement, ast.SwitchStatement):
                new = copy.copy(statement)
                prefix, new.target = self.test_expression(statement.target, scope, stack)
                new.cases = []
                for labels, case in statement.cases:
                    self.check_calls(labels)
                    body = self.block(case.statements, scope.child(), stack=stack, returns=returns, edit=edit and not prefix)
                    new.cases.append((self.rewrite(labels, scope), ast.CompoundStatement(body)))
                if statement.default is not None:
                    new.default = ast.CompoundStatement(self.block(statement.default.statements, scope.child(), stack=stack, returns=returns, edit=edit and not prefix))
                if prefix:
                    new = ast.CompoundStatement(prefix + [new])
                    self.replace_control(statement, new, starting_count, edit=edit, scope_body=scope_body and len(statements) == 1)
            else:
                self.check_calls(statement)
                if isinstance(statement, (ast.ClassicalDeclaration, ast.ConstantDeclaration, ast.IODeclaration)):
                    new = copy.copy(statement)
                    new.type = self.rewrite(statement.type, scope)
                    if hasattr(statement, 'init_expression'):
                        new.init_expression = self.rewrite(statement.init_expression, scope)
                elif isinstance(statement, ast.AliasStatement):
                    new = copy.copy(statement)
                    new.value = self.rewrite(statement.value, scope)
                else:
                    new = self.rewrite(statement, scope, quantum=isinstance(statement, (ast.QuantumGate, ast.QuantumPhase)))
                if isinstance(statement, (ast.ClassicalDeclaration, ast.ConstantDeclaration, ast.IODeclaration)):
                    name = statement.identifier.name
                    value = None
                    if isinstance(statement, ast.ConstantDeclaration):
                        try:
                            value = scope.evaluator().cast(statement.type, scope.evaluator().evaluate(statement.init_expression))
                        except ConstantEvaluationError as exc:
                            value = exc
                    replacement = self.fresh(name) if scope.local else identifier(name)
                    scope.mask(name, replacement)
                    new.identifier = replacement
                    if value is not None:
                        scope.constants[name] = value
                    if global_scope and isinstance(statement, ast.ConstantDeclaration) and name in self.captured_names:
                        proxy = self.fresh('capture')
                        scope.captures[name] = proxy
                        declaration = ast.ConstantDeclaration(copy.deepcopy(statement.type), proxy, identifier(name))
                        if edit:
                            _, end = self.editor.bounds(statement, statement)
                            self.editor.edits.append((end, end, '\n' + self.print_statements([declaration])))
                        self.count += 1
                        self.check_budget(statement)
                elif isinstance(statement, ast.AliasStatement):
                    name = statement.target.name
                    try:
                        ref = self.reference(statement.value, scope)
                    except InliningError:
                        ref = None
                    replacement = self.fresh(name) if scope.local else identifier(name)
                    scope.mask(name, replacement)
                    new.target = replacement
                    if ref is not None:
                        if ref.scalar:
                            # Qiskit's importer does not support scalar aliases;
                            # substitute only fixed selections. Re-evaluating a
                            # mutable index after this declaration changes which
                            # qubit a measurement or reset would address.
                            if scope.local and ref.members is None:
                                raise error(statement, 'scalar quantum aliases require constant indices')
                            scope.renames[name] = as_operand(ref.expression)
                            scope.qubits[name] = ref
                            if scope.local:
                                continue
                        else:
                            scope.qubits[name] = QubitReference(replacement, False, ref.members, ref.bases)
                elif isinstance(statement, ast.QubitDeclaration):
                    if scope.local or not global_scope:
                        raise error(statement, 'qubit declarations must be at global scope')
                    try:
                        size = scope.evaluator().integer(statement.size) if statement.size is not None else 1
                    except ConstantEvaluationError as exc:
                        raise error(statement, f'qubit size must be constant: {exc}') from exc
                    if size < 1:
                        raise error(statement, 'qubit size must be positive')
                    self.qubit_count += size
                    if self.width is not None and self.qubit_count > self.width:
                        raise error(statement, f'circuit has more qubits than --width {self.width}')
                    name = statement.qubit.name
                    scope.mask(name, identifier(name))
                    scope.qubits[name] = QubitReference(identifier(name), statement.size is None,
                                                       tuple((name, i) for i in range(size)), frozenset([name]))
                elif isinstance(statement, (ast.QuantumGateDefinition, ast.ExternDeclaration)):
                    scope.functions[statement.name.name] = statement
                    scope.names.add(statement.name.name)
                elif isinstance(statement, ast.ReturnStatement) and returns is not None:
                    destination, returned, kind = returns
                    if (destination is None) != (statement.expression is None):
                        raise error(statement, 'return value does not match the subroutine return type')
                    if isinstance(statement.expression, ast.QuantumMeasurement) and not isinstance(kind, ast.BitType):
                        raise error(statement, 'measurement returns currently require a bit return type')
                    if destination is not None:
                        result.append(assign(destination, new.expression))
                    result.extend([assign(returned, ast.BooleanLiteral(True)), ast.BreakStatement()])
                    continue
            result.append(new)
        return result


def inline_subroutines(source, program, editor, *, max_inline_statements=DEFAULT_MAX_INLINE_STATEMENTS, width=None):
    if max_inline_statements < 1:
        raise ValueError('max_inline_statements must be positive')
    if not quantum_definitions(program):
        return source, program, False
    # Work on our own AST: attaching literal tokens must not mutate caller data.
    compiler = Inliner(source, copy.deepcopy(program), editor, max_inline_statements, width)
    scope = Scope()
    # Standard gate/function names are handled normally unless the caller
    # introduces a binding with the same spelling.
    for name in ('x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz',
                 'cx', 'cy', 'cz', 'ch', 'swap', 'ccx', 'cswap', 'p', 'cp', 'U',
                 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'pow'):
        scope.functions[name] = name
    compiler.block(compiler.program.statements, scope, edit=True, global_scope=True)
    output = editor.finish(program.version)
    return output, openqasm3.parse(output), True
