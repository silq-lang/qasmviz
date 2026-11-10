import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import openqasm3
from openqasm3 import ast

from qasmviz_dynamic_costs import collect_dynamic_costs
from qasmviz_loops import nodes
from qasmviz_inline import quantum_definitions
from qasmviz_qasm3 import (
    SourceCompilationError, compile_clifford_t_qasm3,
    compile_clifford_t_source, compile_clifford_t_source_with_ast,
)
from subroutine_fixture import execute


HEADER = 'OPENQASM 3.0; include "stdgates.inc";\n'
ROOT = Path(__file__).resolve().parents[1]


class SubroutineTests(unittest.TestCase):
    def equivalent(self, body, *, measurements=0, initial=None, **options):
        source = HEADER + body
        original = openqasm3.parse(source)
        output, compiled = compile_clifford_t_source_with_ast(source, epsilon=1e-9, **options)
        reparsed = openqasm3.parse(output)
        self.assertFalse(quantum_definitions(reparsed))
        for outcomes in itertools.product((0, 1), repeat=measurements):
            expected = execute(original, initial=initial, outcomes=outcomes)
            actual = execute(reparsed, initial=initial, outcomes=outcomes)
            self.assertAlmostEqual(actual[1], expected[1], places=7)
            if expected[1] > 1e-12:
                np.testing.assert_allclose(actual[0], expected[0], atol=1e-7, rtol=0)
                self.assertEqual(actual[2], expected[2])
        return output, compiled

    def test_scalar_register_slice_and_alias_arguments(self):
        body = '''def pair(qubit[2] p) { h p[0]; cx p[0],p[1]; }
def single(qubit p) { t p; }
qubit[3] q; let a=q[1:2]; pair(a); single(q[0]);'''
        self.equivalent(body, initial=np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j]) / np.sqrt(204))

    def test_measurement_return_reset_and_caller_condition(self):
        self.equivalent('''def xmeasure(qubit p) -> bit { h p; bit m=measure p; reset p; return m; }
qubit q; bit m=xmeasure(q); if(m) { x q; }''', measurements=1)

    def test_register_measurement_return(self):
        self.equivalent('''def sample(qubit[2] p) -> bit[2] { h p; return measure p; }
qubit[2] q; bit[2] result=sample(q);''', measurements=2)

    def test_nested_calls_and_return_call(self):
        self.equivalent('''def sample(qubit p) -> bit { h p; return measure p; }
def outer(qubit p) -> bit { return sample(p); }
qubit q; bit m=outer(q); if(m) { x q; }''', measurements=1)

    def test_classical_arguments_are_private_copies(self):
        self.equivalent('''def change(int x, bit b, qubit p) -> int {
x += 5; b=measure p; if(b) { return x; } return x+1;
} qubit q; h q; int x=10; bit b=0; int result=change(x,b,q);''', measurements=1)

    def test_early_return_through_nested_loops_does_not_exit_caller(self):
        self.equivalent('''def early(int choice, qubit p) -> int {
for int i in [0:2] {
  int j=0; while(j<3) {
    if(j==choice) { t p; return i+j; }
    j += 1;
  }
  h p;
}
return 99;
}
qubit q; int total=0; int r;
for int k in [0:2] { r=early(k,q); total += r; h q; }
''')

    def test_void_return_break_and_continue_targets(self):
        self.equivalent('''def helper(qubit p) {
for int i in [0:3] {
  if(i==0) { continue; }
  h p;
  if(i==1) { return; }
  t p;
}
x p;
}
qubit q; int count=0;
for int i in [0:2] { helper(q); count += 1; t q; }
''')

    def test_nested_callee_return_does_not_return_from_caller(self):
        self.equivalent('''def first(qubit p) { t p; return; x p; }
def second(qubit p) { for int i in [0:1] { first(p); h p; } t p; }
qubit q; second(q);''')

    def test_switch_return_and_end(self):
        self.equivalent('''def choose(int v, qubit p) -> int {
switch(v) { case 0 { t p; return 1; } default { h p; return 2; } }
} qubit q; int a=choose(0,q); int b=choose(1,q);''')
        self.equivalent('def stop(qubit p) { h p; end; } qubit q; int n=0; stop(q); n=1; x q;')

    def test_ignored_return_value_still_measures(self):
        self.equivalent('def sample(qubit p)->bit { h p; return measure p; } qubit q; sample(q); t q;', measurements=1)

    def test_returned_variable_remains_in_caller_scope(self):
        self.equivalent('''def one(qubit p)->int { t p; return 1; }
qubit q; int result=0; if(true) { int value=one(q); result=value+1; }''')

    def test_constants_locals_and_repeated_calls_are_hygienic(self):
        self.equivalent('''const float[64] theta=pi/4;
def rotate(float[64] amount, qubit p)->int {
 int temp=1; if(true) { int temp=2; if(temp==2) { rz(theta+amount) p; } }
 return temp;
}
qubit q; int temp=20; int a; int b;
if(true) { const float[64] theta=0; a=rotate(0,q); b=rotate(pi/4,q); }
''')

    def test_builtin_constant_is_resolved_in_defining_scope(self):
        self.equivalent('def f(qubit p) { rz(pi/4) p; } qubit q; if(true) { const float[64] pi=0; f(q); }')

    def test_loop_arguments_keep_scalar_conversion_semantics(self):
        self.equivalent('''def rotate(float[64] v, qubit p) { rz(v/2) p; }
qubit[2] q; for int i in [0:1] { rotate(i,q[i]); }''')

    def test_classical_helper_calls_and_runtime_scalar_returns(self):
        self.equivalent('''def plus(int v)->int { return v+2; }
def sample(qubit p)->bit { h p; return measure p; }
def choose(int v, qubit p)->int { if(v>2) { t p; } return plus(v); }
qubit q; bit m=sample(q); int r=choose(plus(m),q);''', measurements=1)

    def test_conditional_void_call_keeps_its_scope(self):
        self.equivalent('def f(qubit p) { h p; t p; } qubit q; if(false) f(q); x q;')

    def test_named_register_sizes_and_caller_shadowing(self):
        self.equivalent('''const int n=2; def f(qubit[n] p) { t p[1]; }
qubit[2] q; if(true) { const int n=1; f(q); }''')

    def test_constant_parameter_in_local_loop_and_float32_rounding(self):
        self.equivalent('''def f(int n, float[32] a, qubit[2] p) {
for int i in [0:n-1] { rz(a) p[i]; }
} qubit[2] q; f(2,0.3,q);''')

    def test_width_checked_before_expanding_qubit_references(self):
        with self.assertRaisesRegex(SourceCompilationError, '--width 1'):
            compile_clifford_t_source(HEADER + 'def f(qubit p) { h p; } qubit[1000000000] q; f(q[0]);', width=1)

    def test_generated_identifiers_do_not_capture_user_names(self):
        self.equivalent('''def f(qubit p)->int { h p; return 7; }
qubit q; int __qasmviz_result_1=99; int result=f(q); result+=__qasmviz_result_1;''')

    def test_local_alias_and_nested_register_parameter(self):
        self.equivalent('''def single(qubit p) { h p; }
def many(qubit[2] p) { let one=p[0]; single(one); single(p[1]); }
qubit[2] q; many(q);''')

    def test_custom_gate_capture(self):
        source = HEADER + 'const float[64] a=pi/4; gate g p { rz(a) p; } def f(qubit p) { g p; } qubit q; f(q);'
        output = compile_clifford_t_source(source)
        self.assertNotIn('gate g ', output)
        self.assertNotIn('g q;', output)
        self.assertIn('t q;', output)

    def test_source_precision_comments_unicode_and_unbraced_calls(self):
        literal = '0.1234567890123456789012345678901234'
        source = HEADER + f'''// untouched π comment
def f(float[128] arg, qubit p) -> float[128] {{
float[128] local={literal}; h p; return local+arg;
}}
qubit q; float[128] result;
if(true) result=f({literal},q); // unchanged trailing comment
'''
        result = compile_clifford_t_source(source)
        self.assertEqual(result.count(literal), 2)
        self.assertIn('// untouched π comment', result)
        self.assertIn('// unchanged trailing comment', result)
        parsed = openqasm3.parse(result)
        self.assertTrue(any(isinstance(n, ast.BranchingStatement) for n in nodes(parsed)))

    def test_api_does_not_mutate_supplied_ast(self):
        source = HEADER + 'def f(qubit p) { h p; } qubit q; f(q);'
        program = openqasm3.parse(source)
        original = copy.deepcopy(program)
        compile_clifford_t_source_with_ast(source, program=program)
        self.assertEqual(program, original)
        compile_clifford_t_qasm3(program)
        self.assertEqual(program, original)

    def test_imaginary_and_duration_literals_keep_their_tokens(self):
        literal = '1.2345678901234567890123456789012345'
        source = HEADER + f'''def f(qubit p) {{
complex[float[128]] c={literal}im; duration d={literal}ns; h p;
}} qubit q; f(q);'''
        output = compile_clifford_t_source(source)
        self.assertIn(literal + 'im', output)
        self.assertIn(literal + 'ns', output)
        openqasm3.parse(output)

    def test_ast_overflow_literal_reports_source_api_alternative(self):
        source = HEADER + 'def f(qubit p) { float[128] n=1e10000; h p; } qubit q; f(q);'
        self.assertIn('1e10000', compile_clifford_t_source(source))
        with self.assertRaisesRegex(SourceCompilationError, 'use the source API'):
            compile_clifford_t_qasm3(openqasm3.parse(source))

    def test_inlined_quantum_work_is_in_dynamic_costs(self):
        _, program = self.equivalent('''def sample(qubit p)->bit { t p; h p; return measure p; }
qubit q; bit m=sample(q);''', measurements=1, optimization_level=0)
        costs = collect_dynamic_costs(program)
        self.assertEqual(costs['t-count'], {'lower': 1, 'upper': 1})
        self.assertEqual(costs['measurements'], {'lower': 1, 'upper': 1})
        self.assertEqual(costs['gate-count'], {'lower': 2, 'upper': 2})

    def test_errors_are_explicit(self):
        cases = [
            ('def f(qubit p) { f(p); } qubit q; f(q);', 'recursive'),
            ('qubit q; f(q); def f(qubit p) { h p; }', 'before its definition'),
            ('def f(qubit a,qubit b) { cx a,b; } qubit q; f(q,q);', 'overlap'),
            ('def f(qubit[2] p,qubit r) { h p; } qubit[3] q; f(q[0:1],q[1]);', 'overlap'),
            ('def f(qubit a,qubit b) { cx a,b; } qubit[3] q; for int i in [0:1] { f(q[i],q[2]); }', 'cannot prove'),
            ('def f(qubit p) { h p; } qubit[1] q; f(q);', 'scalar'),
            ('def f(qubit[2] p) { h p; } qubit[1] q; f(q);', 'size 2'),
            ('def f(qubit p) { h p; } qubit[2] q; f(q[2]);', 'out of range'),
            ('def f(qubit p) { h p; } qubit q; f();', 'expects 1'),
            ('def f(qubit p)->int { h p; return 1; } qubit q; int n=1+f(q);', 'split this expression'),
            ('def f(qubit p)->int { h p; return 1; } qubit q; int n=0; n+=f(q);', 'require ='),
            ('def f(qubit p) { h p; } qubit q; int n=f(q);', 'void'),
            ('def f(bit b,qubit p)->int { if(b) { h p; return 1; } } qubit q; int n=f(0,q);', 'every path'),
            ('def f(qubit p)->int { h p; return; } qubit q; int n=f(q);', 'return type'),
            ('def f(readonly array[int, 2] a, qubit p) { h p; } array[int,2] a; qubit q; f(a,q);', 'array-reference'),
            ('float[64] theta=0.1; def f(qubit p) { rz(theta) p; } qubit q; f(q);', 'cannot capture'),
            ('qubit q; def f(qubit p) { h q; } qubit p; f(p);', 'cannot capture'),
            ('def f(float[64] a,qubit p) { rz(a) p; } qubit q; float[64] a=0.1; f(a,q);', 'classical evaluation'),
            ('def f(float[64] a,qubit p) { a=0.2; rz(a) p; } qubit q; f(0.1,q);', 'classical evaluation'),
            ('def f(qubit p) { h p; } qubit q; if(true) { int h=0; f(q); }', 'shadows captured'),
            ('def f(qubit p) { if(b) { h p; } } qubit q; if(true) { bit b=1; f(q); }', 'unbound name'),
            ('def f(qubit h) { h h; } qubit q; f(q);', 'shadows callable'),
            ('def f(int i,qubit[2] p) { let a=p[i]; i=1; reset a; } qubit[2] q; f(0,q);', 'aliases require constant'),
        ]
        for body, message in cases:
            with self.subTest(body=body), self.assertRaisesRegex(SourceCompilationError, message):
                compile_clifford_t_source(HEADER + body)

    def test_expansion_budget_is_shared_and_validated(self):
        body = 'def f(qubit p) { h p; } qubit q; f(q); f(q);'
        with self.assertRaisesRegex(SourceCompilationError, 'max-inline-statements'):
            compile_clifford_t_source(HEADER + body, max_inline_statements=5)
        compile_clifford_t_source(HEADER + body, max_inline_statements=100)
        compile_clifford_t_source(HEADER + 'def f(qubit p) { h p; } qubit q; f(q);', max_inline_statements=4)
        with self.assertRaisesRegex(ValueError, 'positive'):
            compile_clifford_t_source(HEADER + body, max_inline_statements=0)

    def test_cli_dump_json_limits_and_no_tracebacks(self):
        source = HEADER + 'def f(qubit p) { t p; } qubit q; f(q);'
        for flags in [('--dump',), ('--json',), ('--dump', '--max-inline-statements', '1')]:
            proc = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), '--clifford-t', *flags],
                                  input=source, text=True, capture_output=True, timeout=30)
            if len(flags) == 3:
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn('max-inline-statements', proc.stderr)
                self.assertNotIn('Traceback', proc.stderr)
            else:
                self.assertEqual(proc.returncode, 0, proc.stderr)
                if flags == ('--json',):
                    self.assertEqual(json.loads(proc.stdout)['t-count'], {'lower': 1, 'upper': 1})
                else:
                    self.assertIn('t q;', proc.stdout)
        for flags in [('--max-inline-statements', '1'), ('--clifford-t', '--max-inline-statements', '0')]:
            proc = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *flags], input=source,
                                  text=True, capture_output=True, timeout=30)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('max-inline-statements', proc.stderr)


if __name__ == '__main__':
    unittest.main()
