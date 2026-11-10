"""Clifford+T source transformation without importing classical code into Qiskit.

This is deliberately a source compiler, not a complete OpenQASM type checker or
interpreter. Classical statements are preserved. Only straight-line unitary
regions with compile-time parameters and operands are sent to Qiskit.
"""

import copy
import math
from bisect import bisect_left
from dataclasses import dataclass, field, fields

from antlr4 import InputStream
import openqasm3
from openqasm3 import ast
from openqasm3.parser import qasm3Lexer
from qiskit.exceptions import QiskitError
from qiskit_qasm3_import import parse as import_quantum
from qiskit_qasm3_import.exceptions import ConversionError

from qasmviz_clifford_t import CLIFFORD_T_BASIS, clifford_t_pass_manager
from qasmviz_constants import ConstantEvaluationError, ConstantEvaluator


class SourceCompilationError(ValueError):
    """A source construct cannot yet be safely lowered by this compiler."""


class _SourceEditor:
    """Edit quantum regions only, retaining classical spelling and precision.

    The reference AST stores floating literals as Python floats. Printing the
    whole AST could therefore round otherwise untouched float[128] constants.
    Statement spans end at a semicolon or closing brace (one-character tokens).
    """

    def __init__(self, source):
        self.source = source
        self.lines = [0] + [i + 1 for i, char in enumerate(source) if char == "\n"]
        self.edits = []
        # Use the parser's lexer so braces inside comments or strings cannot
        # be mistaken for the scope enclosing a single-statement body.
        self.tokens = [token for token in qasm3Lexer(InputStream(source)).getAllTokens()
                       if token.channel == 0]
        self.starts = [token.start for token in self.tokens]

    def bounds(self, first, last):
        return (self.lines[first.span.start_line - 1] + first.span.start_column,
                self.lines[last.span.end_line - 1] + last.span.end_column + 1)

    def replace(self, region, replacement, *, scope_body=False):
        start, end = self.bounds(region[0], region[-1])
        previous = bisect_left(self.starts, start) - 1
        unbraced = scope_body and previous >= 0 and self.tokens[previous].text != "{"
        line = self.source[self.lines[region[0].span.start_line - 1]:start]
        indent = line[:len(line) - len(line.lstrip(" \t"))]
        lines = openqasm3.dumps(ast.Program(replacement)).rstrip().splitlines()
        if unbraced:
            # if (b) x q; must become if (b) { h q; s q; s q; h q; }.
            text = "{\n" + "".join(indent + "  " + line + "\n" for line in lines) + indent + "}"
        else:
            text = ("\n" + indent).join(lines)
        self.edits.append((start, end, text))

    def finish(self, version):
        result, position = [], 0
        for start, end, text in sorted(self.edits):
            if start < position:
                raise SourceCompilationError("overlapping source transformations")
            result.extend((self.source[position:start], text))
            position = end
        result.append(self.source[position:])
        output = "".join(result)
        if version is None:
            output = "OPENQASM 3.0;\n" + output
        return output if output.endswith("\n") else output + "\n"


def _error(node, message):
    span = getattr(node, "span", None)
    location = f"{span.start_line}:{span.start_column + 1}: " if span else ""
    return SourceCompilationError(location + message)


def _nodes(value):
    """Walk nested AST lists and tuples, including switch case bodies."""
    if isinstance(value, ast.QASMNode):
        yield value
        for item in fields(value):
            yield from _nodes(getattr(value, item.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nodes(item)


def needs_source_compilation(program):
    """Keep ordinary static circuits on qasmviz's existing circuit path."""
    circuit_statements = (
        ast.Include, ast.QubitDeclaration, ast.QuantumGateDefinition,
        ast.QuantumGate, ast.QuantumPhase, ast.QuantumReset,
        ast.QuantumBarrier, ast.QuantumMeasurementStatement,
        ast.AliasStatement, ast.DelayInstruction, ast.Pragma,
    )
    for statement in program.statements:
        if isinstance(statement, circuit_statements):
            continue
        if (isinstance(statement, ast.ClassicalDeclaration) and isinstance(statement.type, ast.BitType)
                and (statement.init_expression is None
                     or isinstance(statement.init_expression, ast.QuantumMeasurement))):
            continue
        return True
    return False


@dataclass
class _Scope:
    preamble: list = field(default_factory=list)
    qubit_operands: list = field(default_factory=list)
    quantum_names: set = field(default_factory=set)
    classical_names: set = field(default_factory=set)
    constants: dict = field(default_factory=dict)

    def child(self):
        return _Scope(
            list(self.preamble), list(self.qubit_operands),
            set(self.quantum_names), set(self.classical_names),
            dict(self.constants),
        )


def _quantum_alias(expression, names):
    if isinstance(expression, ast.Identifier):
        return expression.name in names
    if isinstance(expression, ast.IndexExpression):
        return _quantum_alias(expression.collection, names)
    if isinstance(expression, ast.Concatenation):
        return _quantum_alias(expression.lhs, names) and _quantum_alias(expression.rhs, names)
    return False


class _Compiler:
    def __init__(self, optimization_level, epsilon, width, editor=None):
        self.width = width
        self.editor = editor
        self.pm = clifford_t_pass_manager(
            optimization_level=optimization_level,
            epsilon=epsilon,
            qubits_initially_zero=False,
        )

    @staticmethod
    def expression(node, scope, bound=()):
        """Fold scalar expressions, except formal gate parameters.

        Constant subexpressions in a gate definition are frozen in its defining
        scope, before a later call can introduce differently named bindings.
        """
        if any(isinstance(child, ast.Identifier) and child.name in bound for child in _nodes(node)):
            result = copy.deepcopy(node)
            if isinstance(result, ast.BinaryExpression):
                result.lhs = _Compiler.expression(result.lhs, scope, bound)
                result.rhs = _Compiler.expression(result.rhs, scope, bound)
            elif isinstance(result, ast.UnaryExpression):
                result.expression = _Compiler.expression(result.expression, scope, bound)
            elif not isinstance(result, ast.Identifier):
                raise _error(node, 'this expression with a formal gate parameter is not supported')
            return result
        try:
            value = ConstantEvaluator(scope.constants, scope.classical_names).evaluate(node)
            if value.kind == 'bool':
                raise ConstantEvaluationError(node, 'quantum parameters must be numeric constants')
            return value.literal()
        except ConstantEvaluationError as exc:
            raise _error(exc.node, str(exc)) from exc

    @staticmethod
    def index(node, scope):
        if isinstance(node, list):
            return [_Compiler.index(item, scope) for item in node]
        if isinstance(node, ast.RangeDefinition):
            return ast.RangeDefinition(*(
                _Compiler.index(item, scope) if item is not None else None
                for item in (node.start, node.end, node.step)
            ))
        if isinstance(node, ast.DiscreteSet):
            return ast.DiscreteSet([_Compiler.index(item, scope) for item in node.values])
        try:
            return ast.IntegerLiteral(ConstantEvaluator(scope.constants, scope.classical_names).integer(node))
        except ConstantEvaluationError as exc:
            raise _error(exc.node, str(exc)) from exc

    @staticmethod
    def operand(node, scope):
        result = copy.deepcopy(node)
        if isinstance(result, ast.IndexedIdentifier):
            result.indices = _Compiler.index(result.indices, scope)
        elif isinstance(result, ast.IndexExpression):
            result.collection = _Compiler.operand(result.collection, scope)
            result.index = _Compiler.index(result.index, scope)
        elif isinstance(result, ast.Concatenation):
            result.lhs = _Compiler.operand(result.lhs, scope)
            result.rhs = _Compiler.operand(result.rhs, scope)
        return result

    @staticmethod
    def quantum_statement(statement, scope, bound=()):
        result = copy.deepcopy(statement)
        if isinstance(result, ast.QuantumGate):
            result.arguments = [_Compiler.expression(arg, scope, bound) for arg in result.arguments]
        elif isinstance(result, ast.QuantumPhase):
            result.argument = _Compiler.expression(result.argument, scope, bound)
        for modifier in getattr(result, 'modifiers', ()):
            if modifier.argument is not None:
                modifier.argument = _Compiler.expression(modifier.argument, scope, bound)
        result.qubits = [_Compiler.operand(qubit, scope) for qubit in result.qubits]
        return result

    def preflight(self, program):
        for node in _nodes(program):
            if getattr(node, "annotations", None):
                raise _error(node, "annotations are not supported by source Clifford+T compilation")
            if isinstance(node, (ast.CalibrationDefinition, ast.CalibrationStatement,
                                 ast.CalibrationGrammarDeclaration, ast.Box, ast.DurationOf)):
                raise _error(node, "calibrations, boxes and durationof need a target timing model")
            if isinstance(node, ast.Include) and node.filename != "stdgates.inc":
                raise _error(node, "source Clifford+T compilation supports only stdgates.inc includes")
            if isinstance(node, ast.SubroutineDefinition):
                quantum = (ast.QuantumStatement, ast.QuantumMeasurement, ast.QuantumArgument)
                if any(isinstance(child, quantum) for child in _nodes(node)):
                    raise _error(node, "quantum subroutines are not yet supported; inline the subroutine")
            if isinstance(node, ast.Identifier) and node.name.startswith("$"):
                raise _error(node, "physical qubits are not yet supported by source Clifford+T compilation")

    def add_qubits(self, statement, scope):
        folded = copy.deepcopy(statement)
        if folded.size is not None:
            folded.size = self.index(folded.size, scope)
            if folded.size.value < 1:
                raise _error(statement, 'qubit size must be a positive integer constant')
            if self.width is not None and len(scope.qubit_operands) + folded.size.value > self.width:
                raise _error(statement, f'circuit has more qubits than --width {self.width}')
        try:
            circuit = import_quantum(openqasm3.dumps(ast.Program([folded], version="3.0")))
        except (ConversionError, QiskitError) as exc:
            raise _error(statement, f"qubit sizes must be compile-time expressions supported by Qiskit: {exc}") from exc
        name = statement.qubit.name
        if name in scope.quantum_names or name in scope.classical_names:
            raise _error(statement, f"duplicate declaration of {name!r}")
        scope.quantum_names.add(name)
        scope.preamble.append(folded)
        if statement.size is None:
            scope.qubit_operands.append(ast.Identifier(name))
        else:
            scope.qubit_operands.extend(
                ast.IndexedIdentifier(ast.Identifier(name), [[ast.IntegerLiteral(i)]])
                for i in range(circuit.num_qubits)
            )
        if self.width is not None and len(scope.qubit_operands) > self.width:
            raise _error(statement, f"circuit has more qubits than --width {self.width}")

    def compile_region(self, region, scope):
        if not region:
            return []
        if not any(isinstance(node, ast.Include) and node.filename == "stdgates.inc"
                   for node in scope.preamble):
            raise _error(region[0], 'source Clifford+T compilation requires include "stdgates.inc"')
        folded = [self.quantum_statement(statement, scope) for statement in region]
        for node in _nodes(folded):
            if isinstance(node, ast.FloatLiteral) and not math.isfinite(node.value):
                raise _error(node, "quantum parameters must be finite")
            if isinstance(node, ast.Identifier) and node.name in scope.classical_names:
                raise _error(node, f"{node.name!r} in a quantum operation requires classical evaluation; "
                             "use literal parameters and static qubit indices for this initial compiler")
        source = openqasm3.dumps(ast.Program(scope.preamble + folded, version="3.0"))
        try:
            original = import_quantum(source)
            if original.num_qubits != len(scope.qubit_operands):
                raise _error(region[0], "quantum block refers to undeclared qubits")
            if original.parameters:
                raise _error(region[0], "quantum parameters must be known at compile time")
            numbers = [original.global_phase]
            numbers.extend(param for instruction in original.data for param in instruction.operation.params)
            if not all(math.isfinite(float(number)) for number in numbers):
                raise _error(region[0], "quantum parameters must be finite")
            compiled = self.pm.run(original)
        except (ConversionError, QiskitError) as exc:
            raise _error(region[0], "cannot compile quantum block (parameters and indices must be "
                         f"compile-time expressions supported by Qiskit): {exc}") from exc
        return self.emit_region(compiled, scope, region[0])

    @staticmethod
    def emit_region(circuit, scope, source_node):
        """Emit primitives, retaining phase and restoring logical wire order.

        ElidePermutations can remove SWAPs even with qubits_initially_zero=False.
        Its layout is part of the result's semantics. Translate the entry layout
        and explicitly undo the exit permutation before the next source region.
        """
        operands = scope.qubit_operands
        n = len(operands)
        initial = list(range(n))
        final = list(range(n))
        if circuit.layout is not None:
            initial = circuit.layout.initial_index_layout(filter_ancillas=True)
            final = circuit.layout.final_index_layout(filter_ancillas=True)
        if (circuit.num_qubits != n or len(initial) != n or len(final) != n
                or set(initial) != set(range(n)) or set(final) != set(range(n))):
            raise _error(source_node, "backend introduced unsupported extra qubits or layout")
        physical_to_source = [0] * n
        for logical, physical in enumerate(initial):
            physical_to_source[physical] = logical
        result = []

        def gate(name, indices):
            if name in scope.classical_names:
                raise _error(source_node, f"classical name {name!r} shadows an emitted basis gate")
            for i in indices:
                if any(isinstance(node, ast.Identifier) and node.name in scope.classical_names
                       for node in _nodes(operands[i])):
                    raise _error(source_node, "a classical declaration shadows a qubit needed by "
                                 "the compiled block; use distinct names")
            result.append(ast.QuantumGate(
                [], ast.Identifier(name), [], [copy.deepcopy(operands[i]) for i in indices]
            ))

        phase = float(circuit.global_phase)
        if phase != 0:
            result.append(ast.QuantumPhase([], ast.FloatLiteral(phase), []))
        for instruction in circuit.data:
            operation = instruction.operation
            if operation.name not in CLIFFORD_T_BASIS or operation.params or instruction.clbits:
                raise _error(source_node, f"backend left unsupported operation {operation.name!r}")
            gate(operation.name, [physical_to_source[circuit.find_bit(bit).index]
                                  for bit in instruction.qubits])

        positions = [physical_to_source[physical] for physical in final]
        occupants = [0] * n
        for logical, position in enumerate(positions):
            occupants[position] = logical
        for logical in range(n):
            position = positions[logical]
            if position == logical:
                continue
            # Restore a SWAP using only the advertised Clifford+T basis.
            gate("cx", [logical, position])
            gate("cx", [position, logical])
            gate("cx", [logical, position])
            displaced = occupants[logical]
            positions[logical], positions[displaced] = logical, position
            occupants[logical], occupants[position] = logical, displaced
        return result

    def block(self, statements, scope, *, global_scope=False, scope_body=False):
        result, region = [], []

        def flush():
            replacement = self.compile_region(region, scope)
            if region and self.editor is not None:
                self.editor.replace(region, replacement, scope_body=scope_body and len(statements) == 1)
            result.extend(replacement)
            region.clear()

        for statement in statements:
            if isinstance(statement, (ast.QuantumGate, ast.QuantumPhase)):
                region.append(statement)
                continue
            flush()
            if isinstance(statement, ast.Include):
                scope.preamble.append(statement)
            elif isinstance(statement, ast.QubitDeclaration):
                if not global_scope:
                    raise _error(statement, "qubit declarations must be at global scope")
                self.add_qubits(statement, scope)
            elif isinstance(statement, ast.QuantumGateDefinition):
                definition = copy.deepcopy(statement)
                bound = {arg.name for arg in definition.arguments + definition.qubits}
                definition.body = [self.quantum_statement(item, scope, bound) for item in definition.body]
                scope.preamble.append(definition)
                # All calls are lowered in place. Keeping the original definition
                # could leave unused non-Clifford gates in the emitted program.
                if self.editor is not None:
                    self.editor.replace([statement], [])
                continue
            elif isinstance(statement, ast.AliasStatement):
                name = statement.target.name
                if _quantum_alias(statement.value, scope.quantum_names):
                    if name in scope.quantum_names:
                        raise _error(statement, "shadowing a quantum name with an alias is not yet supported")
                    alias = copy.deepcopy(statement)
                    alias.value = self.operand(alias.value, scope)
                    scope.preamble.append(alias)
                    scope.quantum_names.add(name)
                    scope.classical_names.discard(name)
                else:
                    scope.classical_names.add(name)
                    scope.quantum_names.discard(name)
                scope.constants.pop(name, None)
            elif isinstance(statement, ast.ConstantDeclaration):
                name = statement.identifier.name
                evaluator = ConstantEvaluator(scope.constants, scope.classical_names)
                try:
                    value = evaluator.evaluate(statement.init_expression)
                    value = evaluator.cast(statement.type, value)
                except ConstantEvaluationError as exc:
                    # Preserve unsupported classical constants unless a quantum
                    # expression actually needs their values.
                    value = exc
                scope.constants[name] = value
                scope.classical_names.add(name)
                scope.quantum_names.discard(name)
            elif isinstance(statement, (ast.ClassicalDeclaration, ast.IODeclaration)):
                name = statement.identifier.name
                scope.classical_names.add(name)
                scope.quantum_names.discard(name)
                scope.constants.pop(name, None)
            elif isinstance(statement, (ast.SubroutineDefinition, ast.ExternDeclaration)):
                name = statement.name.name
                scope.classical_names.add(name)
                scope.constants.pop(name, None)
            elif isinstance(statement, ast.BranchingStatement):
                statement.if_block = self.block(statement.if_block, scope.child(), scope_body=True)
                statement.else_block = self.block(statement.else_block, scope.child(), scope_body=True)
            elif isinstance(statement, (ast.WhileLoop, ast.ForInLoop)):
                child = scope.child()
                if isinstance(statement, ast.ForInLoop):
                    child.classical_names.add(statement.identifier.name)
                    child.quantum_names.discard(statement.identifier.name)
                    child.constants.pop(statement.identifier.name, None)
                statement.block = self.block(statement.block, child, scope_body=True)
            elif isinstance(statement, ast.SwitchStatement):
                for _, case in statement.cases:
                    case.statements = self.block(case.statements, scope.child())
                if statement.default is not None:
                    statement.default.statements = self.block(statement.default.statements, scope.child())
            elif isinstance(statement, ast.CompoundStatement):
                statement.statements = self.block(statement.statements, scope.child())
            # Measurements, resets, barriers, delays and classical statements
            # retain their original position and form; every one ends a region.
            result.append(statement)
        flush()
        return result


def compile_clifford_t_qasm3(program, *, optimization_level=3, epsilon=None, width=None):
    """Return a new AST; never mutate the caller's parsed program.

    Width is an upper bound. This path does not allocate clean ancillas: a qubit
    entering a region may be live or entangled, even if it is unused locally.
    """
    compiler = _Compiler(optimization_level, epsilon, width)
    compiler.preflight(program)
    result = copy.deepcopy(program)
    result.statements = compiler.block(result.statements, _Scope(), global_scope=True)
    result.version = result.version or "3.0"
    return result


def compile_clifford_t_source(source, *, program=None, optimization_level=3, epsilon=None, width=None):
    """Transform source text without reprinting its classical statements.

    The optional program must be the unmodified AST parsed from this source;
    it lets the CLI avoid parsing twice when selecting a compilation path.
    """
    program = openqasm3.parse(source) if program is None else program
    editor = _SourceEditor(source)
    compiler = _Compiler(optimization_level, epsilon, width, editor)
    compiler.preflight(program)
    compiler.block(copy.deepcopy(program.statements), _Scope(), global_scope=True)
    return editor.finish(program.version)
