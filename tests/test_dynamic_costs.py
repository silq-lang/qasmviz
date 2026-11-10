import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import openqasm3

from qasmviz_dynamic_costs import collect_dynamic_costs
from qasmviz_qasm3 import compile_clifford_t_source_with_ast
from test_loops import HEADER, execute_fixture


ROOT = Path(__file__).resolve().parents[1]


def costs(body):
    _, program = compile_clifford_t_source_with_ast(HEADER + body, optimization_level=0)
    return collect_dynamic_costs(program)


class DynamicCostTests(unittest.TestCase):
    def test_exclusive_branches_do_not_add_their_total_gate_counts(self):
        result = costs('qubit q; bit b; if(b) { t q; } else { h q; }')
        self.assertEqual(result['gate-count'], {'lower': 1, 'upper': 1})
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': 1})
        self.assertIsNone(result['depth'])
        self.assertIsNone(result['t-depth'])

    def test_constant_conditions_and_no_else(self):
        self.assertEqual(costs('qubit q; const bool b=true; if(b) { t q; } else { h q; }')['t-count'],
                         {'lower': 1, 'upper': 1})
        self.assertEqual(costs('qubit q; bit b; if(b) { t q; }')['gate-count'], {'lower': 0, 'upper': 1})

    def test_finite_loop_counts_and_specialized_regions(self):
        result = costs('qubit[3] q; for int i in [0:2] { t q[i]; }')
        self.assertEqual(result['t-count'], {'lower': 3, 'upper': 3})
        self.assertEqual(result['gate-count'], {'lower': 3, 'upper': 3})
        result = costs('qubit q; for int i in {1, 1, 3} { t q; }')
        self.assertEqual(result['t-count'], {'lower': 3, 'upper': 3})

    def test_large_loop_counts_use_arithmetic_not_expansion(self):
        result = costs('qubit q; for int i in [1:1000000000000] { t q; }')
        self.assertEqual(result['t-count'], {'lower': 10**12, 'upper': 10**12})
        result = costs('qubit q; for int i in [1:1000000000000] { t q; break; }')
        self.assertEqual(result['t-count'], {'lower': 1, 'upper': 1})

    def test_break_continue_and_nested_exit_targets(self):
        result = costs('qubit q; for int i in [0:4] { t q; if(i==1) { break; } }')
        self.assertEqual(result['t-count'], {'lower': 2, 'upper': 2})
        result = costs('qubit q; for int i in [0:4] { if(i==1) { continue; } t q; }')
        self.assertEqual(result['t-count'], {'lower': 4, 'upper': 4})
        result = costs('qubit q; for int i in [0:2] { for int j in [0:9] { t q; break; } h q; }')
        self.assertEqual(result['t-count'], {'lower': 3, 'upper': 3})
        self.assertEqual(result['gate-count'], {'lower': 6, 'upper': 6})

    def test_unknown_counts_are_null_not_infinity(self):
        result = costs('qubit q; bit b; while(b) { t q; }')
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': None})
        json.dumps(result, allow_nan=False)
        result = costs('qubit q; int n=3; for int i in [0:n] { h q; }')
        self.assertEqual(result['gate-count'], {'lower': 0, 'upper': None})
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': 0})

    def test_empty_loops_and_unreachable_code(self):
        for body in ('for int i in [2:1] { t q; }', 'while(false) { t q; }', 'end; t q;'):
            with self.subTest(body=body):
                self.assertEqual(costs('qubit q; ' + body)['t-count'], {'lower': 0, 'upper': 0})
        result = costs('end; qubit[2] q; t q[0];')
        self.assertEqual(result['width'], 2)

    def test_end_stops_all_containing_loops(self):
        result = costs('qubit q; for int i in [0:2] { for int j in [0:2] { t q; end; } h q; } t q;')
        self.assertEqual(result['gate-count'], {'lower': 1, 'upper': 1})

    def test_constant_true_while_and_early_break(self):
        result = costs('qubit q; while(true) { h q; } t q;')
        self.assertEqual(result['gate-count'], {'lower': 0, 'upper': None})
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': 0})
        result = costs('qubit q; while(true) { h q; break; } t q;')
        self.assertEqual(result['gate-count'], {'lower': 2, 'upper': 2})

    def test_iterator_writes_prevent_unsound_condition_folding(self):
        result = costs('qubit q; for int i in [0:2] { i=1; if(i==1) { t q; } }')
        self.assertLessEqual(result['t-count']['lower'], 3)
        self.assertGreaterEqual(result['t-count']['upper'], 3)

    def test_measurement_and_reset_broadcasts_slices_and_aliases(self):
        result = costs('const int n=4; qubit[n] q; bit[n] b; b=measure q; let pair=q[1:2]; reset pair;')
        self.assertEqual(result['measurements'], {'lower': 4, 'upper': 4})
        self.assertEqual(result['resets'], {'lower': 2, 'upper': 2})
        result = costs('qubit[4] q; bit[2] b=measure q[{1,3}]; reset q[3:-1:1];')
        self.assertEqual(result['measurements'], {'lower': 2, 'upper': 2})
        self.assertEqual(result['resets'], {'lower': 3, 'upper': 3})

    def test_unknown_register_slice_is_not_reported_as_one_operation(self):
        result = costs('qubit[4] q; int n=2; reset q[0:n];')
        self.assertEqual(result['resets'], {'lower': 0, 'upper': None})

    def test_switch_cases_are_alternatives(self):
        result = costs('qubit q; int a; switch(a) { case 0 { t q; } case 1 { h q; } default { s q; } }')
        self.assertEqual(result['gate-count'], {'lower': 1, 'upper': 1})
        result = costs('qubit q; const int a=1; switch(a) { case 0 { t q; } case 1 { h q; } }')
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': 0})

    def test_classical_calls_may_end_execution(self):
        result = costs('def stop() { end; } qubit q; stop(); t q;')
        self.assertEqual(result['t-count'], {'lower': 0, 'upper': 1})

    def test_bounds_contain_every_enumerated_fixture_path(self):
        fixtures = (
            'if(a) { t q; } if(!a) { h q; }',
            'for int i in [0:3] { if(a) { t q; } else { h q; } if(b) { break; } }',
            'for int i in [0:3] { if(a && i==1) { continue; } t q; if(b && i==2) { end; } } h q;',
            'for int i in [0:2] { for int j in [0:1] { if(a) { break; } t q; } if(b) { continue; } h q; }',
        )
        for fixture in fixtures:
            _, program = compile_clifford_t_source_with_ast(HEADER + 'qubit q; bit a; bit b; ' + fixture,
                                                          optimization_level=0)
            result = collect_dynamic_costs(program)
            for a, b in itertools.product((False, True), repeat=2):
                with self.subTest(fixture=fixture, a=a, b=b):
                    circuit, _ = execute_fixture(program, {'a': a, 'b': b})
                    operations = circuit.count_ops()
                    counts = {'gate-count': sum(operations.values()),
                              't-count': operations.get('t', 0) + operations.get('tdg', 0),
                              'cx-count': operations.get('cx', 0)}
                    for metric, count in counts.items():
                        self.assertLessEqual(result[metric]['lower'], count)
                        if result[metric]['upper'] is not None:
                            self.assertGreaterEqual(result[metric]['upper'], count)


class DynamicCommandLineTests(unittest.TestCase):
    def cli(self, *args, source=''):
        return subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *args], input=source,
                              text=True, capture_output=True, cwd=ROOT, timeout=60)

    def test_json_for_original_example(self):
        result = self.cli('--clifford-t', '--json', str(ROOT / 'tests/circuits/classical-feedforward.qasm'))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['analysis'], 'dynamic')
        self.assertEqual(report['t-count'], {'lower': 0, 'upper': 0})
        self.assertEqual(report['measurements'], {'lower': 1, 'upper': 2})

    def test_cost_dump_and_default_modes(self):
        source = HEADER + 'qubit[2] q; for int i in [0:1] { t q[i]; }'
        for flags in (('--cost',), (), ('--dump', '--cost')):
            with self.subTest(flags=flags):
                result = self.cli('--clifford-t', *flags, source=source)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('t-count: 2', result.stdout)
        result = self.cli('--clifford-t', '--dump', '--no-cost', source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        openqasm3.parse(result.stdout)

    def test_json_cost_combination_emits_only_json(self):
        result = self.cli('--clifford-t', '--json', '--cost', source=HEADER + 'qubit q; bit b; if(b) { t q; }')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['t-count'], {'lower': 0, 'upper': 1})

    def test_mixed_static_and_dynamic_json_is_one_array(self):
        with tempfile.TemporaryDirectory() as directory:
            static, dynamic = Path(directory) / 'static.qasm', Path(directory) / 'dynamic.qasm'
            static.write_text(HEADER + 'qubit q; h q;')
            dynamic.write_text(HEADER + 'qubit q; bit b; if(b) { t q; }')
            result = self.cli('--clifford-t', '--json', str(static), str(dynamic))
            self.assertEqual(result.returncode, 0, result.stderr)
            reports = json.loads(result.stdout)
            self.assertEqual([report['file'] for report in reports], [str(static), str(dynamic)])
            self.assertNotIn('analysis', reports[0])
            self.assertEqual(reports[1]['analysis'], 'dynamic')

    def test_loop_limit_and_validation(self):
        source = HEADER + 'qubit[3] q; for int i in [0:2] { h q[i]; }'
        result = self.cli('--clifford-t', '--dump', '--max-loop-variants', '2', source=source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--max-loop-variants', result.stderr)
        self.assertEqual(result.stdout, '')
        for flags in (('--clifford-t', '--max-loop-variants', '0'), ('--max-loop-variants', '3')):
            result = self.cli(*flags, source=source)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--max-loop-variants', result.stderr)


if __name__ == '__main__':
    unittest.main()
