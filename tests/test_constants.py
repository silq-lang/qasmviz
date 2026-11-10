import copy
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import openqasm3
from openqasm3 import ast
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_qasm3_import import parse as import_quantum

from qasmviz_qasm3 import (
    SourceCompilationError, compile_clifford_t_qasm3, compile_clifford_t_source,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = 'OPENQASM 3.0; include "stdgates.inc";\n'


def emitted_circuit(output, *, sizes=None, otherwise=False):
    """Read emitted primitives independently of the compiler's constant resolver.

    Tests supply register sizes and branch choices explicitly. Original alias
    declarations are unnecessary because emitted gates use the physical names.
    """
    statements = []

    def collect(block):
        for statement in block:
            if isinstance(statement, ast.QubitDeclaration):
                statement = copy.deepcopy(statement)
                if sizes and statement.qubit.name in sizes:
                    statement.size = ast.IntegerLiteral(sizes[statement.qubit.name])
                statements.append(statement)
            elif isinstance(statement, (ast.Include, ast.QuantumGate, ast.QuantumPhase)):
                statements.append(statement)
            elif isinstance(statement, ast.BranchingStatement):
                collect(statement.else_block if otherwise else statement.if_block)
    collect(openqasm3.parse(output).statements)
    return import_quantum(openqasm3.dumps(ast.Program(statements, version='3.0')))


class ConstantCompilerTests(unittest.TestCase):
    def assert_circuit(self, output, expected, **kwargs):
        actual = emitted_circuit(output, **kwargs)
        error = np.linalg.norm(Operator(actual).data - Operator(expected).data, 2)
        self.assertLess(error, 1e-7)

    def test_dynamic_named_size_angle_and_index(self):
        declarations = '''const uint[32] SIZE = 2;
const float[64] theta = pi / 7;
const int index = SIZE - 1;
qubit[SIZE] q;
bit b;
b = measure q[0];
'''
        source = HEADER + declarations + 'if (b) { rz(theta) q[index]; } else { x q[0]; }\n'
        output = compile_clifford_t_source(source, epsilon=1e-9)
        self.assertIn(declarations, output)
        self.assertIn('if (b)', output)
        for otherwise in (False, True):
            expected = QuantumCircuit(2)
            expected.x(0) if otherwise else expected.rz(math.pi / 7, 1)
            self.assert_circuit(output, expected, sizes={'q': 2}, otherwise=otherwise)

    def test_alias_ranges_sets_and_chained_constants(self):
        source = HEADER + '''const int n = 2 + 2;
const int first = 1;
const int last = n - 1;
qubit[n] q;
let pair = q[{first, last}];
let tail = q[last:last];
h pair; cx tail,q[first];
'''
        expected = QuantumCircuit(4)
        expected.h(1)
        expected.h(3)
        expected.cx(3, 1)
        output = compile_clifford_t_source(source)
        self.assertIn('let pair = q[{first, last}];', output)
        self.assert_circuit(output, expected, sizes={'q': 4})

    def test_scopes_and_definition_site_capture(self):
        source = HEADER + '''const float theta = pi / 4;
gate pulse(a) p { rz(theta + a) p; }
qubit q;
bit b;
if (b) { const float theta = pi / 2; rz(theta) q; pulse(pi / 4) q; }
else { const float theta = pi; rz(theta) q; }
rz(theta) q;
'''
        output = compile_clifford_t_source(source, epsilon=1e-9)
        for otherwise in (False, True):
            expected = QuantumCircuit(1)
            if otherwise:
                expected.rz(math.pi, 0)
            else:
                expected.rz(math.pi / 2, 0)
                expected.rz(math.pi / 2, 0)
            expected.rz(math.pi / 4, 0)
            self.assert_circuit(output, expected, otherwise=otherwise)

    def test_alias_uses_its_defining_scope(self):
        source = HEADER + '''const int index = 0;
qubit[2] q; let a=q[index:index]; bit b;
if (b) { const int index=1; x a; }
x q[index];
'''
        self.assert_circuit(compile_clifford_t_source(source), QuantumCircuit(2))

    def test_gate_formals_shadow_constants(self):
        source = HEADER + '''const float a=pi;
gate pulse(a) p { rz(a) p; }
qubit q; pulse(pi/4) q;
'''
        expected = QuantumCircuit(1)
        expected.rz(math.pi / 4, 0)
        self.assert_circuit(compile_clifford_t_source(source), expected)

    def test_signed_integer_division_and_remainder(self):
        source = HEADER + '''const int[64] a = -7;
const int[64] b = 3;
const int[64] quotient = a / b;
const int[64] remainder = a % b;
const float theta = float(quotient + remainder) / 4;
qubit q; rz(theta) q;
'''
        expected = QuantumCircuit(1)
        expected.rz(-0.75, 0)
        self.assert_circuit(compile_clifford_t_source(source, epsilon=1e-9), expected)

    def test_large_integers_do_not_pass_through_float(self):
        source = HEADER + '''const int[128] a=9007199254740993;
const int index=a-9007199254740992;
qubit[2] q; x q[index];
'''
        expected = QuantumCircuit(2)
        expected.x(1)
        self.assert_circuit(compile_clifford_t_source(source), expected)

    def test_float32_arithmetic_rounds_before_widening(self):
        source = HEADER + '''const uint bits = 32;
const float[bits] a = 16777216;
const float[bits] one = 1;
const float[bits] sum = a + one;
const float[64] theta = float[64](sum) - float[64](a);
qubit q; rz(theta) q;
'''
        self.assert_circuit(compile_clifford_t_source(source), QuantumCircuit(1))

    def test_declared_float32_precision(self):
        source = HEADER + 'const float[32] theta = pi; qubit q; rz(theta) q;'
        expected = QuantumCircuit(1)
        expected.rz(float(np.float32(math.pi)), 0)
        self.assert_circuit(compile_clifford_t_source(source, epsilon=1e-9), expected)

    def test_builtin_math_and_explicit_casts(self):
        source = HEADER + '''const int n = int(ceiling(sqrt(2.0)));
const float theta = arccos(0.0) + log(exp(0.0)) + mod(3.5, 1.0);
const uint[8] bits = 3;
const uint[8] index = (bits >> 1) & uint[8](1);
qubit[n] q; rz(theta) q[index];
'''
        expected = QuantumCircuit(2)
        expected.rz(math.pi / 2 + 0.5, 1)
        self.assert_circuit(compile_clifford_t_source(source, epsilon=1e-9), expected, sizes={'q': 2})

    def test_boolean_constant_and_gate_modifier(self):
        source = HEADER + '''const bool enabled = (1 < 2) && !false;
const int controls = int(enabled);
const float phase = pi / 4;
qubit[2] q;
ctrl(controls) @ x q[0],q[1];
gphase(phase);
'''
        expected = QuantumCircuit(2, global_phase=math.pi / 4)
        expected.cx(0, 1)
        self.assert_circuit(compile_clifford_t_source(source), expected)

    def test_unused_unsupported_classical_constants_are_preserved(self):
        declarations = '''const float[128] precise = 1.000000000000000000000000000001;
const angle[8] phase = pi / 3;
const complex[float[64]] z = 1.0 + 2.0im;
const duration time = 3ns;
'''
        output = compile_clifford_t_source(HEADER + declarations + 'qubit q; h q;')
        self.assertIn(declarations, output)
        expected = QuantumCircuit(1)
        expected.h(0)
        self.assert_circuit(output, expected)

    def test_mutable_bindings_and_loop_variables_mask_constants(self):
        for body in ('if (true) { float a=0.5; rz(a) q; }',
                     'int n=1; for int a in [0:n] { rz(a) q; }'):
            with self.subTest(body=body), self.assertRaisesRegex(SourceCompilationError, 'classical evaluation'):
                compile_clifford_t_source(HEADER + 'const float a=pi; qubit q; ' + body)

    def test_invalid_or_unsupported_quantum_constants_fail(self):
        cases = [
            ('const float[128] a=0.2;', 'float\\[32\\] and float\\[64\\]'),
            ('const angle a=pi/4;', 'AngleType'),
            ('const float a=1e10000;', 'finite'),
            ('const float a=sqrt(-1.0);', 'invalid constant expression'),
            ('const int a=1/0;', 'division by zero'),
            ('const int a=0.5;', 'explicit cast'),
            ('const int bits=0; const int[bits] a=1;', 'constant width'),
            ('const int[8] a=128;', 'overflow'),
            ('const uint a=-1;', 'overflow'),
            ('const uint[8] b=255; const uint[8] a=b+uint[8](1);', 'overflow'),
            ('const int a=2**1000000;', 'too large'),
            ('const float a=sin(1.0, 2.0);', 'arity'),
            ('float runtime=0.2; const float a=runtime;', 'classical evaluation'),
            ('const float a=later; const float later=0.2;', 'classical evaluation'),
            ('const bool a=false && runtime;', 'classical evaluation'),
            ('extern f(float) -> float; const float a=f(0.2);', 'not a built-in'),
        ]
        for declarations, message in cases:
            with self.subTest(declarations=declarations), self.assertRaisesRegex(SourceCompilationError, message):
                compile_clifford_t_source(HEADER + declarations + 'qubit q; rz(a) q;')

    def test_sizes_and_indices_require_integers_and_width_is_enforced(self):
        for body, message, options in (
            ('const int n=0; qubit[n] q;', 'positive integer', {}),
            ('const float n=2.0; qubit[n] q;', 'integer constant', {}),
            ('const int n=1000000; qubit[n] q;', '--width', {'width': 2}),
            ('const float i=1.0; qubit[2] q; h q[i];', 'integer constant', {}),
        ):
            with self.subTest(body=body), self.assertRaisesRegex(SourceCompilationError, message):
                compile_clifford_t_source(HEADER + body, **options)

    def test_ast_is_not_mutated(self):
        program = openqasm3.parse(HEADER + 'const int n=2; qubit[n] q; x q[n-1];')
        before = copy.deepcopy(program)
        compile_clifford_t_qasm3(program)
        self.assertEqual(program, before)

    def test_cli_accepts_constants_and_reports_errors_without_tracebacks(self):
        for declarations, success in (('const float a=pi/7;', True), ('const int a=1/0;', False)):
            result = subprocess.run(
                [sys.executable, str(ROOT / 'qasmviz.py'), '--clifford-t', '--dump', '--eps', '1e-6'],
                input=HEADER + declarations + '\nqubit q; rz(a) q;',
                capture_output=True, text=True, cwd=ROOT, timeout=60,
            )
            if success:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(declarations, result.stdout)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('stdin:3:', result.stderr)
                self.assertIn('division by zero', result.stderr)
                self.assertEqual(result.stdout, '')
            self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
