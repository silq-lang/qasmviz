import math
import operator
import unittest

import numpy as np
import openqasm3
from openqasm3 import ast
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qasmviz_qasm3 import SourceCompilationError, compile_clifford_t_source


HEADER = 'OPENQASM 3.0; include "stdgates.inc";\n'


def execute_fixture(program, initial=None):
    """Execute only these test fixtures; independent of compiler evaluators.

    Quantum operations are appended to a reference circuit, so comparison is
    on every input state, including relative and global phases.
    """
    values = {'pi': math.pi, **(initial or {})}
    wires = {}
    circuit = QuantumCircuit()

    class Break(Exception):
        pass

    class Continue(Exception):
        pass

    class End(Exception):
        pass

    def value(node):
        if isinstance(node, (ast.IntegerLiteral, ast.FloatLiteral, ast.BooleanLiteral)):
            return node.value
        if isinstance(node, ast.Identifier):
            return values[node.name]
        if isinstance(node, ast.UnaryExpression):
            return {'!': operator.not_, '-': operator.neg}[node.op.name](value(node.expression))
        if isinstance(node, ast.BinaryExpression):
            return {'+': operator.add, '-': operator.sub, '*': operator.mul,
                    '==': operator.eq, '!=': operator.ne, '<': operator.lt,
                    '<=': operator.le, '>': operator.gt, '>=': operator.ge,
                    '&&': lambda a, b: a and b, '||': lambda a, b: a or b,
                    '/': operator.truediv}[node.op.name](value(node.lhs), value(node.rhs))
        raise AssertionError(f'unexpected fixture expression: {node}')

    def operand(node):
        if isinstance(node, ast.Identifier):
            return wires[node.name][0]
        return wires[node.name.name][value(node.indices[0][0])]

    def block(statements):
        for statement in statements:
            if isinstance(statement, ast.QubitDeclaration):
                count = value(statement.size) if statement.size else 1
                offset = circuit.num_qubits
                from qiskit.circuit import Qubit
                circuit.add_bits([Qubit() for _ in range(count)])
                wires[statement.qubit.name] = list(range(offset, offset + count))
            elif isinstance(statement, (ast.ConstantDeclaration, ast.ClassicalDeclaration)):
                if statement.init_expression is not None:
                    values[statement.identifier.name] = value(statement.init_expression)
                else:
                    values.setdefault(statement.identifier.name, False)
            elif isinstance(statement, ast.ClassicalAssignment):
                name = statement.lvalue.name
                rhs = value(statement.rvalue)
                values[name] = rhs if statement.op.name == '=' else {
                    '+=': operator.add, '-=': operator.sub,
                }[statement.op.name](values[name], rhs)
            elif isinstance(statement, ast.QuantumGate):
                getattr(circuit, statement.name.name)(*(value(arg) for arg in statement.arguments),
                                                      *(operand(q) for q in statement.qubits))
            elif isinstance(statement, ast.QuantumPhase):
                circuit.global_phase += value(statement.argument)
            elif isinstance(statement, ast.BranchingStatement):
                block(statement.if_block if value(statement.condition) else statement.else_block)
            elif isinstance(statement, ast.ForInLoop):
                domain = statement.set_declaration
                if isinstance(domain, ast.RangeDefinition):
                    start, end = value(domain.start), value(domain.end)
                    step = value(domain.step) if domain.step else 1
                    sequence = range(start, end + (1 if step > 0 else -1), step)
                else:
                    sequence = [value(item) for item in domain.values]
                name = statement.identifier.name
                previous = values.get(name)
                for item in sequence:
                    values[name] = item
                    try:
                        block(statement.block)
                    except Continue:
                        continue
                    except Break:
                        break
                if previous is None:
                    values.pop(name, None)
                else:
                    values[name] = previous
            elif isinstance(statement, ast.BreakStatement):
                raise Break
            elif isinstance(statement, ast.ContinueStatement):
                raise Continue
            elif isinstance(statement, ast.EndStatement):
                raise End
            elif isinstance(statement, ast.CompoundStatement):
                block(statement.statements)
            elif not isinstance(statement, ast.Include):
                raise AssertionError(f'unexpected fixture statement: {statement}')
    try:
        block(program.statements)
    except End:
        pass
    return circuit, values


class LoopCompilerTests(unittest.TestCase):
    def equivalent(self, source, *, initial=None, **options):
        before, expected_values = execute_fixture(openqasm3.parse(HEADER + source), initial)
        output = compile_clifford_t_source(HEADER + source, epsilon=1e-9, **options)
        after, actual_values = execute_fixture(openqasm3.parse(output), initial)
        self.assertLess(np.linalg.norm(Operator(before).data - Operator(after).data, 2), 1e-7)
        self.assertEqual(actual_values, expected_values)
        return output

    def test_indexed_range_at_all_optimization_levels(self):
        for level in range(4):
            with self.subTest(level=level):
                output = self.equivalent('qubit[3] q; for int i in [0:2] { h q[i]; }', optimization_level=level)
                self.assertIn('for int i in [0:2]', output)
                self.assertNotIn('h q[i]', output)

    def test_descending_and_discrete_iterations(self):
        self.equivalent('qubit[4] q; for int i in [3:-2:0] { t q[i]; }')
        self.equivalent('qubit[3] q; for uint[8] i in {2, 0, 2} { h q[i]; }')

    def test_loop_parameter_and_outer_constant_shadowing(self):
        self.equivalent('const int i=10; qubit q; for int i in {1, 2} { rz(i*pi/7) q; } rz(i*pi/7) q;')

    def test_nested_independent_loops(self):
        self.equivalent('qubit[4] q; for int i in [0:1] { for int j in [2:3] { cx q[i],q[j]; } }')

    def test_break_continue_and_classical_updates_are_preserved(self):
        source = '''qubit[3] q; int counter=0;
for int i in [0:2] {
  counter += 1;
  if (i == 0) { continue; }
  h q[i];
  if (i == 1) { break; }
}
if (counter == 2) { x q[0]; }
'''
        output = self.equivalent(source)
        self.assertIn('counter += 1;', output)
        self.assertIn('continue;', output)
        self.assertIn('break;', output)

    def test_end_inside_loop_stays_a_program_exit(self):
        self.equivalent('qubit[3] q; for int i in [0:2] { h q[i]; if(i==1) { end; } } x q[2];')

    def test_single_statement_loop_bodies_and_comments(self):
        output = self.equivalent('qubit[3] q; for int i in [0:2] /* { comment */ x q[i];')
        self.assertIn('/* { comment */', output)

    def test_empty_range_and_large_invariant_loop(self):
        output = self.equivalent('qubit q; for int i in [1:0] { rz(i*pi/7) q; }')
        self.assertNotIn('rz(', output)
        # No expansion is needed when the quantum block does not use i.
        output = compile_clifford_t_source(HEADER + 'qubit q; for int i in [0:1000000] { h q; }', max_loop_variants=1)
        self.assertIn('h q;', output)

    def test_classical_literal_precision_survives_specialization(self):
        classical = 'float[128] x = 1.00000000000000000000000000000001;'
        output = compile_clifford_t_source(HEADER + 'qubit[2] q; for int i in [0:1] { ' + classical + ' h q[i]; }')
        self.assertIn(classical, output)
        self.assertEqual(output.count(classical), 1)

    def test_modified_iterator_is_never_specialized(self):
        for modification in ('i += 1;', 'if (true) { i=1; }', 'i[0] = false;'):
            with self.subTest(modification=modification), self.assertRaisesRegex(SourceCompilationError, 'iterator is modified'):
                compile_clifford_t_source(HEADER + 'qubit[3] q; for int i in [0:1] { ' + modification + ' h q[i]; }')

    def test_runtime_iteration_bounds_are_rejected_when_needed(self):
        with self.assertRaisesRegex(SourceCompilationError, 'cannot specialize loop variable'):
            compile_clifford_t_source(HEADER + 'int n=2; qubit[3] q; for int i in [0:n] { h q[i]; }')

    def test_expansion_budget_is_shared_between_regions_and_nested_loops(self):
        cases = (
            ('qubit[3] q; for int i in [0:2] { h q[i]; }', 2),
            ('qubit[2] q; for int i in [0:1] { h q[i]; barrier q; t q[i]; }', 3),
            ('qubit[4] q; for int i in [0:1] { for int j in [2:3] { cx q[i],q[j]; } }', 3),
        )
        for source, limit in cases:
            with self.subTest(source=source), self.assertRaisesRegex(SourceCompilationError, '--max-loop-variants'):
                compile_clifford_t_source(HEADER + source, max_loop_variants=limit)

    def test_raised_limit_does_not_create_a_deep_conditional_chain(self):
        output = compile_clifford_t_source(HEADER + 'qubit[2] q; for int i in [0:299] { h q[i%2]; }',
                                          max_loop_variants=300)
        # Parsing the generated source must work at ordinary recursion limits.
        program = openqasm3.parse(output)
        circuit, _ = execute_fixture(program)
        np.testing.assert_allclose(Operator(circuit).data, np.eye(4), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
