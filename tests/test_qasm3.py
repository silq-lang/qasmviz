import itertools
import json
import operator
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import openqasm3
from openqasm3 import ast
from qiskit.quantum_info import Operator
from qiskit_qasm3_import import parse as import_quantum

from qasmviz_clifford_t import CLIFFORD_T_BASIS
from qasmviz_qasm3 import (
    SourceCompilationError, compile_clifford_t_qasm3, compile_clifford_t_source,
    needs_source_compilation,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (ROOT / "tests/circuits/classical-feedforward.qasm").read_text()
HEADER = 'OPENQASM 3.0; include "stdgates.inc"; '


def transform(source, **kwargs):
    return compile_clifford_t_qasm3(openqasm3.parse(source), **kwargs)


def quantum_only(program):
    """Construct a unitary reference independently from the compiler's emitter."""
    quantum = (ast.Include, ast.QubitDeclaration, ast.QuantumGateDefinition,
               ast.QuantumGate, ast.QuantumPhase, ast.AliasStatement)
    return import_quantum(openqasm3.dumps(ast.Program(
        [s for s in program.statements if isinstance(s, quantum)], version="3.0"
    )))


def branch_circuit(program):
    prefix = [s for s in program.statements if isinstance(
        s, (ast.Include, ast.QubitDeclaration, ast.QuantumGateDefinition, ast.AliasStatement)
    )]
    branch = next(s for s in program.statements if isinstance(s, ast.BranchingStatement))
    return import_quantum(openqasm3.dumps(ast.Program(prefix + branch.if_block, version="3.0")))


def run_example(program):
    """Enumerate this fixture's measurement branches using state-vector algebra.

    Intentionally only implements the fixture's operations. It does not use
    the new compiler to execute or interpret either the source or the result.
    Missing classical values raise KeyError, catching newly introduced reads
    of conditionally uninitialized variables.
    """
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrices = {
        "h": h, "x": np.array([[0, 1], [1, 0]]),
        "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
        "t": np.diag([1, np.exp(1j * np.pi / 4)]),
        "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
    }

    def value(expr, env):
        if isinstance(expr, (ast.IntegerLiteral, ast.BooleanLiteral, ast.FloatLiteral)):
            return expr.value
        if isinstance(expr, ast.Identifier):
            return env[expr.name]
        if isinstance(expr, ast.BinaryExpression):
            operation = {"<": operator.lt, "==": operator.eq}[expr.op.name]
            return operation(value(expr.lhs, env), value(expr.rhs, env))
        if isinstance(expr, ast.UnaryExpression) and expr.op.name == "!":
            return not value(expr.expression, env)
        raise AssertionError(f"unexpected fixture expression {expr}")

    def execute(statements, paths):
        for statement in statements:
            output = []
            for probability, state, env in paths:
                if isinstance(statement, ast.QuantumGate):
                    state = matrices[statement.name.name] @ state
                elif isinstance(statement, ast.QuantumPhase):
                    state = np.exp(1j * value(statement.argument, env)) * state
                elif isinstance(statement, ast.QuantumMeasurementStatement):
                    for outcome in (0, 1):
                        weight = abs(state[outcome]) ** 2
                        if weight > 1e-14:
                            projected = np.zeros(2, complex)
                            projected[outcome] = state[outcome] / np.sqrt(weight)
                            updated = dict(env, **{statement.target.name: bool(outcome)})
                            output.append((probability * weight, projected, updated))
                    continue
                elif isinstance(statement, ast.QuantumReset):
                    state = np.array([1, 0], complex)
                elif isinstance(statement, ast.ClassicalAssignment):
                    assert statement.op.name == "="
                    env = dict(env, **{statement.lvalue.name: value(statement.rvalue, env)})
                elif isinstance(statement, ast.BranchingStatement):
                    body = statement.if_block if value(statement.condition, env) else statement.else_block
                    output.extend(execute(body, [(probability, state, env)]))
                    continue
                elif not isinstance(statement, (ast.Include, ast.ClassicalDeclaration, ast.QubitDeclaration)):
                    raise AssertionError(f"unexpected fixture statement {statement}")
                output.append((probability, state, env))
            paths = output
        return paths

    return execute(program.statements, [(1.0, np.array([1, 0], complex), {})])


class SourceCompilerTests(unittest.TestCase):
    def test_user_example_all_branches(self):
        original = openqasm3.parse(EXAMPLE)
        original_text = openqasm3.dumps(original)
        compiled = compile_clifford_t_qasm3(original)
        self.assertEqual(openqasm3.dumps(original), original_text)
        output = openqasm3.dumps(compiled)
        self.assertIn("int[64] v0;", output)
        self.assertIn("int[64] v1;", output)
        self.assertNotIn("\nt ", output)
        self.assertNotIn("\ntdg ", output)
        paths = run_example(compiled)
        self.assertEqual(len(paths), 3)
        expected = [
            (0.5, np.array([1, 1]) / np.sqrt(2), {"m0": False, "v1": 3}),
            (0.25, np.array([0, 1]), {"m0": True, "m1": False, "v0": 2, "v1": 2}),
            (0.25, np.array([1, 1]) / np.sqrt(2), {"m0": True, "m1": True, "v0": 3, "v1": 3}),
        ]
        for actual, reference, before in zip(paths, expected, run_example(original)):
            p, state, env = actual
            self.assertAlmostEqual(p, reference[0])
            np.testing.assert_allclose(state, reference[1], atol=1e-12)
            self.assertEqual(env, reference[2])
            self.assertEqual(env, before[2])

    def test_numeric_rotations_on_arbitrary_input(self):
        source = HEADER + 'qubit[2] q; int counter; rx(pi/7) q[0]; cx q[0],q[1]; rz(0.234) q[1];'
        for level in range(4):
            with self.subTest(level=level):
                compiled = quantum_only(transform(source, optimization_level=level, epsilon=1e-6))
                original = quantum_only(openqasm3.parse(source))
                self.assertLess(np.linalg.norm(Operator(compiled).data - Operator(original).data, 2), 1e-5)
                self.assertLessEqual(set(compiled.count_ops()), set(CLIFFORD_T_BASIS))

    def test_conditional_rotation_retains_operator_and_phase(self):
        source = HEADER + '''qubit[2] q; int[64] v=3;
        if (2 < v) { rz(pi/7) q[0]; cx q[0],q[1]; p(0.123) q[1]; }
        '''
        original = branch_circuit(openqasm3.parse(source))
        output = openqasm3.parse(compile_clifford_t_source(source, epsilon=1e-7))
        compiled = branch_circuit(output)
        self.assertLess(np.linalg.norm(Operator(compiled).data - Operator(original).data, 2), 1e-6)
        self.assertLessEqual(set(compiled.count_ops()), set(CLIFFORD_T_BASIS))

    def test_conditional_swap_permutations_are_restored(self):
        for level, permutation in itertools.product(range(4), itertools.permutations(range(3))):
            swaps = ' '.join(f'swap q[{i}], q[{j}];' for i, j in enumerate(permutation))
            source = HEADER + f'qubit[3] q; bit b; if(b) {{ h q[0]; {swaps} cx q[0],q[1]; }}'
            with self.subTest(level=level, permutation=permutation):
                # Omit same-wire swaps, which are invalid quantum instructions.
                source = source.replace('swap q[0], q[0];', '').replace('swap q[1], q[1];', '').replace('swap q[2], q[2];', '')
                original = branch_circuit(openqasm3.parse(source))
                compiled = branch_circuit(transform(source, optimization_level=level))
                np.testing.assert_allclose(Operator(compiled).data, Operator(original).data, atol=1e-12)

    def test_broadcast_aliases_and_scalar_names(self):
        source = HEADER + '''
        qubit[2] left; qubit middle; qubit[2] right; int v;
        let a = left[0:1];
        h a; cx a,right; swap middle,right[1];
        '''
        original = quantum_only(openqasm3.parse(source))
        compiled = quantum_only(transform(source))
        np.testing.assert_allclose(Operator(compiled).data, Operator(original).data, atol=1e-12)

    def test_custom_gate_control_preserves_relative_phase(self):
        source = HEADER + '''
        gate phaseflip(a) q { gphase(a); x q; }
        qubit[2] q; int v;
        ctrl @ phaseflip(pi/4) q[0],q[1];
        inv @ s q[1];
        '''
        compiled_ast = transform(source, epsilon=1e-7)
        self.assertFalse(any(isinstance(s, ast.QuantumGateDefinition) for s in compiled_ast.statements))
        np.testing.assert_allclose(
            Operator(quantum_only(compiled_ast)).data,
            Operator(quantum_only(openqasm3.parse(source))).data, atol=1e-6,
        )

    def test_phase_survives_emission_and_reparse(self):
        source = HEADER + 'qubit q; int v; gphase(0.2); p(0.123) q;'
        compiled = openqasm3.parse(openqasm3.dumps(transform(source, epsilon=1e-7)))
        self.assertTrue(any(isinstance(s, ast.QuantumPhase) for s in compiled.statements))
        np.testing.assert_allclose(
            Operator(quantum_only(compiled)).data,
            Operator(quantum_only(openqasm3.parse(source))).data, atol=1e-6,
        )

    def test_classical_loop_and_switch_structure_survives(self):
        source = HEADER + '''
        qubit q; int[64] v = 0;
        while (v < 4) { v += 1; x q; if (v == 2) { continue; } }
        for int i in [0:3] { v += i; x q; }
        switch (v) { case 1, 2 { x q; } default { x q; } }
        '''
        compiled = transform(source)
        original = openqasm3.parse(source)

        def remove_gates(value):
            from dataclasses import fields
            if isinstance(value, ast.QASMNode):
                for field in fields(value):
                    setattr(value, field.name, remove_gates(getattr(value, field.name)))
            elif isinstance(value, list):
                return [remove_gates(item) for item in value if not isinstance(item, (ast.QuantumGate, ast.QuantumPhase))]
            elif isinstance(value, tuple):
                return tuple(remove_gates(item) for item in value)
            return value

        self.assertEqual(openqasm3.dumps(remove_gates(original)), openqasm3.dumps(remove_gates(compiled)))

    def test_measurement_reset_and_barrier_are_boundaries(self):
        source = HEADER + 'qubit q; bit b; int v; h q; barrier q; h q; b=measure q; reset q; x q;'
        compiled = transform(source)
        significant = [type(s) for s in compiled.statements if isinstance(
            s, (ast.QuantumGate, ast.QuantumBarrier, ast.QuantumMeasurementStatement, ast.QuantumReset)
        )]
        self.assertEqual(significant[:6], [ast.QuantumGate, ast.QuantumBarrier, ast.QuantumGate,
                                          ast.QuantumMeasurementStatement, ast.QuantumReset, ast.QuantumGate])

    def test_width_is_checked_without_allocating_ancillas(self):
        source = HEADER + 'qubit[3] q; int v; ccx q[0],q[1],q[2];'
        with self.assertRaisesRegex(SourceCompilationError, '--width'):
            transform(source, width=2)
        compiled = quantum_only(transform(source, width=8))
        self.assertEqual(compiled.num_qubits, 3)
        np.testing.assert_allclose(
            Operator(compiled).data, Operator(quantum_only(openqasm3.parse(source))).data, atol=1e-12,
        )

    def test_classical_only_subroutine_is_preserved(self):
        source = HEADER + 'def bump(int x) -> int { return x+1; } qubit q; int v=1; v=bump(v); x q;'
        self.assertIn('def bump(int x) -> int', openqasm3.dumps(transform(source)))

    def test_unsupported_constructs_fail_explicitly(self):
        cases = [
            ('qubit q; float[64] a=0.2; rz(a) q;', 'classical evaluation'),
            ('qubit[2] q; for int i in [0:1] { x q[i]; }', 'static qubit indices'),
            ('qubit q; if(true) { int q=1; x q; }', 'classical evaluation'),
            ('qubit[1] q; let a=q; if(true) { int q=1; x a; }', 'shadows a qubit'),
            ('qubit q; if(true) { int h=1; x q; }', 'shadows an emitted basis gate'),
            ('qubit q; def f(qubit p) { x p; } f(q);', 'quantum subroutines'),
            ('qubit q; box[1us] { x q; }', 'timing model'),
            ('qubit q; duration d=durationof({x q;});', 'timing model'),
            ('defcalgrammar "openpulse"; cal { }', 'timing model'),
            ('qubit q; @custom\nx q;', 'annotations'),
            ('include "custom.inc"; qubit q;', 'only stdgates.inc'),
            ('int v; x $0;', 'physical qubits'),
            ('qubit q; rz(1e10000) q;', 'finite'),
        ]
        for source, message in cases:
            with self.subTest(source=source), self.assertRaisesRegex(SourceCompilationError, message):
                transform(HEADER + source)

    def test_static_circuit_routing(self):
        for body in ('qubit q; h q;', 'qubit[2] q; let a=q; h a;', 'qubit q; bit b; b=measure q;'):
            self.assertFalse(needs_source_compilation(openqasm3.parse(HEADER + body)))
        for body in ('qubit q; bit b=true; h q;', 'qubit q; bit b; if(b) { x q; }', 'qubit q; int v; x q;'):
            self.assertTrue(needs_source_compilation(openqasm3.parse(HEADER + body)))

    def test_source_preserves_classical_literal_precision_and_comments(self):
        classical = '''// Keep the following decimal literal exactly.
float[128] v = 1.00000000000000000000000000000001;
int[128] n = 170141183460469231731687303715884105727;
'''
        source = HEADER + '\n' + classical + 'qubit q; if (n > 0) { x q; } // trailing comment\n'
        output = compile_clifford_t_source(source)
        self.assertIn(classical, output)
        self.assertIn('// trailing comment', output)
        self.assertNotIn('x q;', output)
        openqasm3.parse(output)

    def test_source_handles_single_statement_scopes(self):
        for statement in (
            'if (b) x q; else x q;',
            'if (b) /* { misleading brace */ x q; else /* comment */ x q;',
            'if (b) if (c) x q; else x q;',
            'while (b) x q;',
            'for int i in [0:1] x q;',
            'if (b) id q; else x q;',
        ):
            with self.subTest(statement=statement):
                source = HEADER + 'qubit q; bit b; bit c; ' + statement
                expected = transform(source, optimization_level=0)
                actual = openqasm3.parse(compile_clifford_t_source(source, optimization_level=0))
                self.assertEqual(openqasm3.dumps(actual), openqasm3.dumps(expected))

    def test_source_handles_unicode_crlf_and_gate_definitions(self):
        source = (HEADER + '\n' + '''gate flip p { x p; }
qubit ψ; int v=0;
if(v==0) { flip ψ; } else flip ψ;
''').replace('\n', '\r\n')
        actual = openqasm3.parse(compile_clifford_t_source(source, optimization_level=0))
        self.assertEqual(openqasm3.dumps(actual), openqasm3.dumps(transform(source, optimization_level=0)))


class CommandLineTests(unittest.TestCase):
    def cli(self, *args, source=EXAMPLE):
        return subprocess.run(
            [sys.executable, str(ROOT / 'qasmviz.py'), *args], input=source,
            capture_output=True, text=True, cwd=ROOT, timeout=60,
        )

    def test_example_from_stdin(self):
        result = self.cli('--clifford-t', '--dump')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(run_example(openqasm3.parse(result.stdout))), 3)

    def test_example_from_file(self):
        result = self.cli('--clifford-t', '--dump', str(ROOT / 'tests/circuits/classical-feedforward.qasm'), source='')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('int[64] v1;', result.stdout)

    def test_dynamic_bit_condition_with_rotation(self):
        result = self.cli('--clifford-t', '--dump', '--eps', '1e-5',
                          source=HEADER + 'qubit q; bit b; b=measure q; if(b) { rz(pi/7) q; }')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('rz(', result.stdout)
        self.assertIn('if(b)', result.stdout)

    def test_unsupported_outputs_are_actionable(self):
        for option in ('--cost', '--run', '--show', '--dump2'):
            with self.subTest(option=option):
                result = self.cli('--clifford-t', '--dump', option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('supports --dump only', result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertNotIn('Traceback', result.stderr)

    def test_runtime_angle_reports_source_location(self):
        result = self.cli('--clifford-t', '--dump', source=HEADER + '\nqubit q; float a=0.2; rz(a) q;')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stdin:2:', result.stderr)
        self.assertIn('classical evaluation', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_invalid_epsilon_is_rejected(self):
        for epsilon in ('nan', 'inf', '-0.1', '1.1'):
            result = self.cli('--clifford-t', '--dump', '--eps', epsilon)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--eps must be finite', result.stderr)

    def test_existing_static_outputs(self):
        source = HEADER + 'qubit[2] q; h q[0]; cx q[0],q[1];'
        for target in ('--clifford-t', '--cx-u', '--cx-sx'):
            with self.subTest(target=target):
                result = self.cli(target, '--json', source=source)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)
        result = self.cli('--run', '--json', source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(result.stdout)['statevector']
        self.assertEqual({entry['basis'] for entry in state}, {'00', '11'})
        result = self.cli('--clifford-t', '--dump2', source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith('OPENQASM 2.0;'))
        result = self.cli('--clifford-t', '--dump', source='OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; x q[0];')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixed_static_dynamic_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'static.qasm'
            second = Path(directory) / 'dynamic.qasm'
            first.write_text(HEADER + 'qubit q; x q;')
            second.write_text(EXAMPLE)
            result = self.cli('--clifford-t', '--dump', str(first), str(second), source='')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('### '), 2)
            self.assertEqual(result.stdout.count('OPENQASM 3.0;'), 2)


if __name__ == '__main__':
    unittest.main()
