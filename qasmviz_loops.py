"""Finite iteration spaces shared by source compilation and cost analysis."""

from dataclasses import dataclass, fields

from openqasm3 import ast

from qasmviz_constants import ConstantEvaluationError, ConstantEvaluator, Scalar


@dataclass
class Iterations:
    count: int
    type: ast.ClassicalType
    evaluator: ConstantEvaluator
    start: int = 0
    step: int = 1
    values: tuple | None = None

    def at(self, index):
        value = (self.values[index] if self.values is not None
                 else Scalar(self.start + self.step * index, 'int'))
        return self.evaluator.cast(self.type, value)


def iterations(loop, constants, classical_names=()):
    evaluator = ConstantEvaluator(constants, classical_names)
    domain = loop.set_declaration
    if isinstance(domain, ast.RangeDefinition):
        if domain.start is None or domain.end is None:
            raise ConstantEvaluationError(domain, 'loop range must have both endpoints')
        start = evaluator.integer(domain.start)
        end = evaluator.integer(domain.end)
        step = evaluator.integer(domain.step) if domain.step is not None else 1
        if step == 0:
            raise ConstantEvaluationError(domain, 'loop range step must not be zero')
        count = 0 if (end - start) * step < 0 else abs(end - start) // abs(step) + 1
        result = Iterations(count, loop.type, evaluator, start, step)
        if count:
            result.at(0)
            result.at(count - 1)
        return result
    if isinstance(domain, ast.DiscreteSet):
        values = tuple(evaluator.evaluate(item) for item in domain.values)
        result = Iterations(len(values), loop.type, evaluator, values=values)
        for index in range(len(values)):
            result.at(index)
        return result
    raise ConstantEvaluationError(domain, 'loop values are not a constant range or discrete set')


def nodes(value):
    if isinstance(value, ast.QASMNode):
        yield value
        for field in fields(value):
            yield from nodes(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from nodes(item)


def writes_name(statements, name):
    """Conservatively detect writes, including indexed assignment targets.

    A same-named binding in a nested scope can produce a false positive; it
    must never cause us to specialize a mutable value as if it were constant.
    """
    for node in nodes(statements):
        if isinstance(node, ast.ClassicalAssignment):
            target = node.lvalue
        elif isinstance(node, ast.QuantumMeasurementStatement):
            target = node.target
        else:
            continue
        # An index is read to choose the destination; only the indexed object
        # is written. Treating m[i] as a write to i prevents valid loops from
        # being specialized and can also invalidate immutable arguments.
        if isinstance(target, ast.IndexedIdentifier):
            target = target.name
        if isinstance(target, ast.Identifier) and target.name == name:
            return True
    return False
