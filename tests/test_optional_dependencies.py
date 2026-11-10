from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OptionalDependencyTests(unittest.TestCase):
    def test_help_static_dump_and_dynamic_compile_without_optional_imports(self):
        # Block the packages before importing qasmviz, in a fresh interpreter.
        script = '''
import importlib.abc
import runpy
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('qblaze', 'qiskit_ibm_runtime', 'cirq', 'pytket'):
            raise AssertionError('unexpected optional import: ' + fullname)

sys.meta_path.insert(0, BlockOptional())
sys.argv = ['qasmviz.py', *sys.argv[1:]]
runpy.run_path('qasmviz.py', run_name='__main__')
'''
        cases = [
            (['--help'], '', '--clifford-t'),
            (['--dump'], 'include "stdgates.inc"; qubit q; h q;', 'h '),
            (['--clifford-t', '--dump'],
             'include "stdgates.inc"; def f(qubit p) { t p; } qubit q; f(q);', 't q;'),
        ]
        for args, source, expected in cases:
            with self.subTest(args=args, source=source):
                result = subprocess.run([sys.executable, '-c', script, *args], cwd=ROOT,
                                        input=source, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)


if __name__ == '__main__':
    unittest.main()
