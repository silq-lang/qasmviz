from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest

from qasmviz_dynamic_costs import print_dynamic_costs


ROOT = Path(__file__).resolve().parents[1]
HEADING = 'Expected execution counts (qubits initially zero):\n'


def render(costs):
    output = StringIO()
    with redirect_stdout(output):
        print_dynamic_costs(costs)
    return output.getvalue()


class ExpectedOutputTests(unittest.TestCase):
    def test_mixed_numbers_preserve_integer_precision(self):
        for numerator, denominator, text in (
            (291, 2, '145 + 1/2'),
            (3, 4, '0 + 3/4'),
            (4, 3, '1 + 1/3'),
            (3, 1, '3'),
            (0, 1, '0'),
            (3 * 10**40 + 1, 3, f'{10**40} + 1/3'),
        ):
            with self.subTest(numerator=numerator, denominator=denominator):
                costs = {
                    'width': 1, 't-count': {'lower': 0, 'upper': None},
                    'expected-counts': {'t-count': {'numerator': numerator, 'denominator': denominator}},
                }
                self.assertIn(HEADING + f'  t-count: {text}\n', render(costs))

    def test_only_informative_expectations_are_displayed(self):
        costs = {
            'width': 1,
            'gate-count': {'lower': 1, 'upper': 2},
            't-count': {'lower': 0, 'upper': None},
            'measurements': {'lower': 1, 'upper': 1},
            'resets': {'lower': 0, 'upper': 0},
            'expected-counts': {
                'gate-count': {'numerator': 3, 'denominator': 2},
                't-count': None,
                'measurements': {'numerator': 1, 'denominator': 1},
                'resets': {'numerator': 0, 'denominator': 1},
                'status': 'partial', 'initial-state': 'zero',
                'reason': 'some gate costs are unknown',
            },
        }
        before = deepcopy(costs)
        output = render(costs)
        self.assertEqual(output.split(HEADING)[1],
                         '  gate-count: 1 + 1/2\n'
                         'Expected-count analysis stopped: some gate costs are unknown.\n'
                         'Depth is not estimated.\n')
        self.assertIn('  measurements: 1\n', output)  # Still present in the bounds.
        self.assertNotIn('exact where known', output)
        self.assertEqual(costs, before)  # Rendering leaves the structured report intact.

    def test_redundant_or_unavailable_expectations_leave_no_empty_section(self):
        for expected in (
            None,
            {'gate-count': {'numerator': 2, 'denominator': 1}, 't-count': None,
             'reason': 'measurement expression limit reached'},
            {'gate-count': {'numerator': 2, 'denominator': 1}},
        ):
            costs = {
                'width': 1,
                'gate-count': {'lower': 2, 'upper': 2},
                't-count': {'lower': 0, 'upper': None},
            }
            if expected is not None:
                costs['expected-counts'] = expected
            with self.subTest(expected=expected):
                output = render(costs)
                self.assertNotIn('Expected', output)
                self.assertIn('  gate-count: 2\n', output)

    def test_zero_expectation_refines_a_nontrivial_bound(self):
        costs = {'width': 1, 't-count': {'lower': 0, 'upper': 1},
                 'expected-counts': {'t-count': {'numerator': 0, 'denominator': 1}}}
        self.assertIn(HEADING + '  t-count: 0\n', render(costs))

    def test_cli_text_is_concise_and_json_keeps_numeric_fractions(self):
        source = 'include "stdgates.inc"; qubit q; bit b; h q; b=measure q; if(b) { t q; }'
        for mode in ([], ['--clifford-t']):
            for output in ('--cost', '--json'):
                with self.subTest(mode=mode, output=output):
                    result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *mode, output],
                                            input=source, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if output == '--json':
                        expected = json.loads(result.stdout)['expected-counts']
                        self.assertEqual(expected['t-count'], {'numerator': 1, 'denominator': 2})
                        self.assertEqual(expected['measurements'], {'numerator': 1, 'denominator': 1})
                        self.assertEqual(expected['initial-state'], 'zero')
                    else:
                        section = result.stdout.split(HEADING)[1]
                        self.assertIn('  gate-count: 1 + 1/2\n', section)
                        self.assertIn('  t-count: 0 + 1/2\n', section)
                        self.assertNotIn('measurements:', section)
                        self.assertNotIn('resets:', section)

    def test_cli_fixed_costs_omit_the_expectation_section(self):
        source = ('include "stdgates.inc"; qubit q; bit b; h q; b=measure q; '
                  'if(b) { x q; } else { x q; }')
        for mode in ([], ['--clifford-t']):
            result = subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *mode, '--cost'],
                                    input=source, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Execution-count bounds', result.stdout)
            self.assertNotIn('Expected', result.stdout)


if __name__ == '__main__':
    unittest.main()
