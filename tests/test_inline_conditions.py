import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import openqasm3

from qasmviz_dynamic_costs import collect_dynamic_costs
from qasmviz_qasm3 import SourceCompilationError, compile_clifford_t_source_with_ast
from subroutine_fixture import execute


HEADER = 'OPENQASM 3.0; include "stdgates.inc";\n'
UPDATE = 'def update(mutable array[int,1] a)->int { a[0]=1; return 0; }\n'
SAMPLE = 'def sample(qubit p)->bit { return measure p; }\n'
COIN = 'def coin(qubit p)->bit { h p; bit b=measure p; reset p; return b; }\n'
ROOT = Path(__file__).resolve().parents[1]


class InlineConditionTests(unittest.TestCase):
    def equivalent(self, body, *, measurements=0, outcomes=None, initial=None, **options):
        source = HEADER + body
        original = openqasm3.parse(source)
        output, compiled = compile_clifford_t_source_with_ast(source, epsilon=1e-9, **options)
        reparsed = openqasm3.parse(output)
        paths = outcomes if outcomes is not None else itertools.product((0, 1), repeat=measurements)
        for path in paths:
            expected = execute(original, initial=initial, outcomes=path)
            actual = execute(reparsed, initial=initial, outcomes=path)
            self.assertAlmostEqual(actual[1], expected[1], places=7)
            if expected[1] > 1e-12:
                np.testing.assert_allclose(actual[0], expected[0], atol=1e-7, rtol=0)
                self.assertEqual(actual[2], expected[2])
        return output, compiled

    def test_original_rejected_helper_is_now_supported(self):
        self.equivalent('qubit q; def f(qubit p) { x p; } f(q);', initial=[0.6, 0.8j])

    def test_classical_argument_snapshot_survives_later_argument_side_effect(self):
        self.equivalent(UPDATE + SAMPLE + '''
def forward(int i, qubit[2] p, int ignored)->bit { return sample(p[i]); }
qubit[2] q; array[int,1] a={0}; x q[1];
bit result=forward(a[0], q, update(a));
''', measurements=1)

    def test_dynamic_qubit_selection_before_side_effect_is_rejected(self):
        body = UPDATE + '''def read(qubit p, int ignored)->bit { return measure p; }
qubit[2] q; array[int,1] a={0}; bit result=read(q[a[0]], update(a));'''
        with self.assertRaisesRegex(SourceCompilationError, 'later argument'):
            compile_clifford_t_source_with_ast(HEADER + body)

    def test_fixed_qubit_selection_and_argument_order_are_preserved(self):
        self.equivalent(UPDATE + '''
def read(qubit p, int ignored)->bit { return measure p; }
qubit[2] q; array[int,1] a={0}; x q[1]; bit result=read(q[0], update(a));
''', measurements=1)

    def test_dynamic_qubit_selection_after_side_effect_is_preserved(self):
        self.equivalent(UPDATE + '''
def read(int ignored, qubit p)->bit { return measure p; }
qubit[2] q; array[int,1] a={0}; x q[1]; bit result=read(update(a), q[a[0]]);
''', measurements=1)

    def test_direct_if_condition_and_unbraced_else(self):
        self.equivalent(COIN + 'qubit q; int result=0; if(coin(q)) result=1; else result=2;', measurements=1)

    def test_nested_condition_calls_have_separate_scopes(self):
        self.equivalent(COIN + '''qubit[2] q; int result=0;
if(coin(q[0])) { if(coin(q[1])) { result=1; } else { result=2; } }
''', outcomes=[(0,), (1, 0), (1, 1)])

    def test_condition_calls_inside_bounded_loop(self):
        self.equivalent(COIN + '''qubit[2] q; int count=0;
for int i in [0:1] { if(coin(q[i])) { count+=1; t q[i]; } }
''', measurements=2)

    def test_while_condition_is_evaluated_on_every_iteration(self):
        _, program = self.equivalent(COIN + '''qubit[2] q; h q[1]; int count=0;
while(coin(q[0])) { count+=1; t q[1]; }
''', outcomes=[(0,), (1, 0), (1, 1, 0)])
        costs = collect_dynamic_costs(program)
        self.assertIsNone(costs['measurements']['upper'])
        self.assertIsNone(costs['t-count']['upper'])

    def test_continue_rechecks_condition(self):
        self.equivalent(COIN + '''qubit[2] q; h q[1]; int count=0;
while(coin(q[0])) { count+=1; if(count==1) { continue; } t q[1]; }
''', outcomes=[(0,), (1, 0), (1, 1, 0), (1, 1, 1, 0)])

    def test_break_does_not_recheck_condition(self):
        self.equivalent(COIN + '''qubit[2] q; h q[1]; int count=0;
while(coin(q[0])) { count+=1; t q[1]; break; }
''', measurements=1)

    def test_negation_and_casts_preserve_condition_value(self):
        self.equivalent(COIN + '''qubit q; int count=0;
while(!bool(coin(q))) { count+=1; if(count==2) { break; } }
''', outcomes=[(1,), (0, 1), (0, 0)])

    def test_while_condition_uses_current_caller_values(self):
        self.equivalent('''def again(int n,qubit p)->bool { t p; return n<3; }
qubit q; int n=0;
while(again(n,q)) { n+=1; if(true) { int n=99; } }
''', initial=[0.6, 0.8j])

    def test_early_return_in_lowered_loop_exits_its_own_function(self):
        self.equivalent(COIN + '''def choose(qubit p)->int {
while(coin(p)) { return 5; }
return 7;
}
qubit q; int result=0;
for int i in [0:1] { int v=choose(q); result+=v; }
''', measurements=2)

    def test_return_in_if_condition_body_does_not_exit_caller_loop(self):
        self.equivalent(COIN + '''def choose(qubit p)->int {
if(coin(p)) { return 5; } else { return 7; }
}
qubit q; int result=0;
for int i in [0:1] { int v=choose(q); result+=v; }
''', measurements=2)

    def test_end_in_condition_or_body_stops_program(self):
        self.equivalent('def stop(qubit p)->bool { h p; end; } qubit q; int n=0; if(stop(q)) { n=1; } n=2;')
        self.equivalent(COIN + 'qubit q; int n=0; while(coin(q)) { end; } n=1;', measurements=1)

    def test_switch_target_call_and_cast(self):
        self.equivalent(COIN + '''qubit[2] q; h q[1]; int result=0;
switch(int[8](coin(q[0]))) {
case 0 { result=2; h q[1]; }
case 1 { result=3; t q[1]; }
default { result=4; x q[1]; }
}
''', measurements=1)

    def test_condition_body_call_is_inlined_without_overlapping_edits(self):
        self.equivalent(COIN + '''def turn(qubit p) { t p; }
qubit[2] q; h q[1]; if(coin(q[0])) turn(q[1]); else h q[1];
''', measurements=1)

    def test_unmodified_loop_iterator_expressions_remain_specializable(self):
        self.equivalent('''def turn(float[64] v, qubit p) { rz(v) p; }
qubit[2] q; h q; for int i in [0:1] { turn(i*pi/4,q[i]); }
''')

    def test_condition_preserves_classical_literal_precision(self):
        literal = '0.1234567890123456789012345678901234'
        source = HEADER + COIN + f'''qubit q;
// keep this outside comment
if(coin(q)) {{ float[128] a={literal}; }}
while(coin(q)) {{ float[128] a={literal}; break; }}
switch(int(coin(q))) {{ case 0 {{ float[128] a={literal}; }} }}
'''
        output, _ = compile_clifford_t_source_with_ast(source)
        self.assertEqual(output.count(literal), 3)
        self.assertIn('// keep this outside comment', output)
        openqasm3.parse(output)

    def test_condition_expansions_share_budget(self):
        source = HEADER + COIN + 'qubit q; if(coin(q)) { while(coin(q)) { break; } }'
        with self.assertRaisesRegex(SourceCompilationError, 'max-inline-statements'):
            compile_clifford_t_source_with_ast(source, max_inline_statements=10)
        compile_clifford_t_source_with_ast(source, max_inline_statements=100)

    def test_unsupported_expressions_are_rejected_without_reordering(self):
        for test in ('coin(q)==true', 'true && coin(q)', 'coin(q) || true', 'bool(int(coin(q))+1)'):
            with self.subTest(test=test), self.assertRaisesRegex(SourceCompilationError, 'split this expression'):
                compile_clifford_t_source_with_ast(HEADER + COIN + f'qubit q; if({test}) {{ t q; }}')
        for statement in ('if(stop(q)) { t q; }', 'while(stop(q)) { t q; }', 'switch(stop(q)) { case 0 { t q; } }'):
            with self.subTest(statement=statement), self.assertRaisesRegex(SourceCompilationError, 'void'):
                compile_clifford_t_source_with_ast(HEADER + 'def stop(qubit p) { h p; } qubit q; ' + statement)

    def test_if_costs_include_condition_measurement(self):
        _, program = compile_clifford_t_source_with_ast(HEADER + COIN + 'qubit q; if(coin(q)) { t q; }', optimization_level=0)
        costs = collect_dynamic_costs(program)
        self.assertEqual(costs['measurements'], {'lower': 1, 'upper': 1})
        self.assertEqual(costs['resets'], {'lower': 1, 'upper': 1})
        self.assertEqual(costs['t-count'], {'lower': 0, 'upper': 1})

    def test_cli_condition_dump_json_and_diagnostic(self):
        source = HEADER + COIN + 'qubit q; if(coin(q)) { t q; }'
        for option in ('--dump', '--json'):
            proc = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), '--clifford-t', option],
                                  input=source, text=True, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            if option == '--dump':
                self.assertNotIn('coin(', proc.stdout)
                openqasm3.parse(proc.stdout)
            else:
                self.assertEqual(json.loads(proc.stdout)['measurements'], {'lower': 1, 'upper': 1})
        source = HEADER + COIN + 'qubit q; if(coin(q)==true) { t q; }'
        proc = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), '--clifford-t', '--dump'],
                              input=source, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('split this expression', proc.stderr)
        self.assertNotIn('Traceback', proc.stderr)


if __name__ == '__main__':
    unittest.main()
