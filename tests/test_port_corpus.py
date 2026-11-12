import unittest

from run_port_corpus import MANIFEST, run_case


class PortCorpusTests(unittest.TestCase):
    def test_shared_conformance_cases(self):
        self.assertEqual(MANIFEST['schema_version'], 1)
        for case in MANIFEST['cases']:
            with self.subTest(case=case['name']):
                run_case(case)


if __name__ == '__main__':
    unittest.main()
