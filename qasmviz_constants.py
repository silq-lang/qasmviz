"""Evaluate the scalar constants needed by quantum regions.

This is intentionally a subset of OpenQASM constant evaluation. It supports
bool, int, uint and IEEE float[32]/float[64], with float defaulting to 64 bits.
Unsized integers retain exact values; sized integer overflow is rejected rather
than guessing target-specific behavior. Angle, bit-array, complex and duration
values are left to the classical program unless quantum compilation needs them.
"""

import math
import operator
import struct
from dataclasses import dataclass

from openqasm3 import ast


class ConstantEvaluationError(ValueError):
    def __init__(self, node, message):
        super().__init__(message)
        self.node = node


@dataclass(frozen=True)
class Scalar:
    value: int | float | bool
    kind: str
    width: int | None = None

    def literal(self):
        cls = {'bool': ast.BooleanLiteral, 'float': ast.FloatLiteral}.get(self.kind, ast.IntegerLiteral)
        return cls(self.value)


_BUILTINS = dict(zip(('pi', 'π', 'tau', 'τ', 'euler', 'ℇ'),
                     (math.pi, math.pi, math.tau, math.tau, math.e, math.e)))
_FUNCTIONS = {
    'arccos': math.acos, 'arcsin': math.asin, 'arctan': math.atan,
    'ceiling': math.ceil, 'floor': math.floor, 'cos': math.cos,
    'exp': math.exp, 'log': math.log, 'sin': math.sin,
    'sqrt': math.sqrt, 'tan': math.tan,
}
_COMPARE = {'==': operator.eq, '!=': operator.ne, '<': operator.lt,
            '<=': operator.le, '>': operator.gt, '>=': operator.ge}
_ARITHMETIC = {'+': operator.add, '-': operator.sub, '*': operator.mul,
               '**': operator.pow, '&': operator.and_, '|': operator.or_, '^': operator.xor}
_MAX_BITS = 4096


def _checked(node, value, kind, width=None):
    if kind == 'bool':
        return Scalar(bool(value), kind)
    if kind == 'float':
        value = float(value)
        if width == 32:
            value = struct.unpack('!f', struct.pack('!f', value))[0]
        if not math.isfinite(value):
            raise ConstantEvaluationError(node, 'quantum constants must be finite')
        return Scalar(value, kind, width or 64)
    if abs(value).bit_length() > _MAX_BITS:
        raise ConstantEvaluationError(node, f'constant arithmetic exceeds {_MAX_BITS} bits')
    if kind == 'uint' and value < 0:
        raise ConstantEvaluationError(node, 'unsigned constant overflow is not supported')
    if width is not None:
        low, high = (0, 2 ** width) if kind == 'uint' else (-2 ** (width - 1), 2 ** (width - 1))
        if not low <= value < high:
            raise ConstantEvaluationError(node, 'sized integer constant overflow is not supported')
    return Scalar(value, kind, width)


class ConstantEvaluator:
    def __init__(self, bindings, classical_names=()):
        self.bindings = bindings
        self.classical_names = classical_names

    def evaluate(self, node):
        try:
            return self._evaluate(node)
        except ConstantEvaluationError:
            raise
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise ConstantEvaluationError(node, f'invalid constant expression: {exc}') from exc

    def _evaluate(self, node):
        if isinstance(node, ast.BooleanLiteral):
            return Scalar(node.value, 'bool')
        if isinstance(node, ast.IntegerLiteral):
            return _checked(node, node.value, 'int')
        if isinstance(node, ast.FloatLiteral):
            return _checked(node, node.value, 'float', 64)
        if isinstance(node, ast.Identifier):
            if node.name in self.bindings:
                value = self.bindings[node.name]
                if isinstance(value, ConstantEvaluationError):
                    raise ConstantEvaluationError(node, f"cannot resolve const {node.name!r}: {value}")
                return value
            if node.name not in self.classical_names and node.name in _BUILTINS:
                return Scalar(_BUILTINS[node.name], 'float', 64)
            raise ConstantEvaluationError(node, f"{node.name!r} requires classical evaluation; "
                                          'use const parameters and static qubit indices')
        if isinstance(node, ast.Cast):
            return self.cast(node.type, self.evaluate(node.argument), explicit=True)
        if isinstance(node, ast.UnaryExpression):
            value = self.evaluate(node.expression)
            if node.op.name == '!':
                return Scalar(not value.value, 'bool')
            if node.op.name == '-' and value.kind in ('int', 'float'):
                return _checked(node, -value.value, value.kind, value.width)
            if node.op.name == '~' and value.kind == 'uint' and value.width is not None:
                return Scalar((2 ** value.width - 1) ^ value.value, 'uint', value.width)
            raise ConstantEvaluationError(node, f'unsupported constant unary operation {node.op.name!r}')
        if isinstance(node, ast.BinaryExpression):
            # Resolve both operands: even false && runtime_value is not const.
            return self.binary(node, node.op.name, self.evaluate(node.lhs), self.evaluate(node.rhs))
        if isinstance(node, ast.FunctionCall):
            name = node.name.name
            if name in self.classical_names or name in self.bindings:
                raise ConstantEvaluationError(node, f'function {name!r} is not a built-in constant function')
            values = [self.evaluate(argument) for argument in node.arguments]
            if name in ('pow', 'mod') and len(values) == 2:
                return self.binary(node, '**' if name == 'pow' else '%', *values,
                                   float_mod=name == 'mod')
            if name in _FUNCTIONS and len(values) == 1:
                value = values[0]
                width = value.width if value.kind == 'float' else 64
                value = _checked(node, value.value, 'float', width)
                return _checked(node, _FUNCTIONS[name](value.value), 'float', width)
            raise ConstantEvaluationError(node, f'unsupported constant function or arity: {name}')
        raise ConstantEvaluationError(node, f'unsupported constant expression: {type(node).__name__}')

    def integer(self, node):
        value = self.evaluate(node)
        if value.kind not in ('int', 'uint'):
            raise ConstantEvaluationError(node, 'expected an integer constant')
        return value.value

    def cast(self, type_node, value, *, explicit=False):
        kinds = {ast.BoolType: 'bool', ast.IntType: 'int', ast.UintType: 'uint', ast.FloatType: 'float'}
        kind = kinds.get(type(type_node))
        if kind is None:
            raise ConstantEvaluationError(type_node, f'constant type {type(type_node).__name__} is not supported in quantum expressions')
        size = getattr(type_node, 'size', None)
        width = self.integer(size) if size is not None else None
        if width is not None and not 1 <= width <= _MAX_BITS:
            raise ConstantEvaluationError(type_node, f'constant width must be between 1 and {_MAX_BITS}')
        if kind == 'float' and width not in (None, 32, 64):
            raise ConstantEvaluationError(type_node, 'only float[32] and float[64] constants are supported in quantum expressions')
        if kind in ('int', 'uint'):
            if value.kind == 'float' and not explicit:
                raise ConstantEvaluationError(type_node, 'float-to-integer constant conversion requires an explicit cast')
            raw = int(value.value)  # OpenQASM/C truncation towards zero.
        else:
            raw = value.value
        try:
            return _checked(type_node, raw, kind, width)
        except (ArithmeticError, ValueError) as exc:
            if isinstance(exc, ConstantEvaluationError):
                raise
            raise ConstantEvaluationError(type_node, f'invalid constant conversion: {exc}') from exc

    @staticmethod
    def binary(node, op, left, right, *, float_mod=False):
        if op in ('&&', '||'):
            return Scalar(bool(left.value and right.value) if op == '&&'
                          else bool(left.value or right.value), 'bool')
        if left.kind == 'float' or right.kind == 'float':
            kind = 'float'
            width = max(v.width for v in (left, right) if v.kind == 'float')
        else:
            kind = 'uint' if 'uint' in (left.kind, right.kind) else 'int'
            width = max((v.width for v in (left, right) if v.width is not None), default=None)
            if kind == 'uint' and any(v.kind == 'int' and v.value < 0 for v in (left, right)):
                raise ConstantEvaluationError(node, 'mixed signed/unsigned constants with negative values require an explicit cast')
        a = _checked(node, left.value, kind, width).value
        b = _checked(node, right.value, kind, width).value
        if op in _COMPARE:
            return Scalar(_COMPARE[op](a, b), 'bool')
        if op in ('&', '|', '^', '<<', '>>'):
            if left.kind != 'uint' or left.width is None:
                raise ConstantEvaluationError(node, 'bitwise constant operations require sized uint operands')
            if op in ('<<', '>>'):
                if right.kind not in ('int', 'uint') or b < 0:
                    raise ConstantEvaluationError(node, 'constant shift distance must be a nonnegative integer')
                raw = 0 if b >= left.width else ((a << b) & (2 ** left.width - 1) if op == '<<' else a >> b)
                return Scalar(raw, 'uint', left.width)
            if right.kind != 'uint' or right.width != left.width:
                raise ConstantEvaluationError(node, 'bitwise constants must have matching uint widths')
        if op in ('/', '%'):
            if b == 0:
                raise ConstantEvaluationError(node, 'division by zero in constant expression')
            if kind == 'float':
                if op == '%' and not float_mod:
                    raise ConstantEvaluationError(node, 'the % operator requires integer constants')
                raw = a / b if op == '/' else math.fmod(a, b)
            else:
                quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                raw = quotient if op == '/' else a - quotient * b
        elif op in _ARITHMETIC:
            if op == '**' and kind != 'float':
                if b < 0 or b > _MAX_BITS or abs(a).bit_length() * b > _MAX_BITS:
                    raise ConstantEvaluationError(node, 'integer constant exponent is negative or too large')
            raw = _ARITHMETIC[op](a, b)
        else:
            raise ConstantEvaluationError(node, f'unsupported constant binary operation {op!r}')
        return _checked(node, raw, kind, width)
