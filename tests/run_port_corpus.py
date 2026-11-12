"""Run the shared fixture manifest against a CLI supplied as positional arguments.

Run the Python baseline with ``python tests/run_port_corpus.py``, or another
implementation with ``python tests/run_port_corpus.py -- /path/to/qasmviz``.
The JSON expectations contain numeric reference results, not SDK objects.
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Operator, Statevector
from qiskit_qasm3_import import parse


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'tests/conformance'
MANIFEST = json.loads((CORPUS / 'manifest.json').read_text())
DEFAULT_COMMAND = [sys.executable, str(ROOT / 'qasmviz.py')]


def assert_subset(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, f'missing JSON key: {key}'
            assert_subset(actual[key], value)
    else:
        assert actual == expected, f'{actual!r} != {expected!r}'


def run_case(case, command=None):
    expected = case['expect']
    process = subprocess.run([*(command or DEFAULT_COMMAND), *case['args']],
                             input=(CORPUS / case['source']).read_text(),
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == expected.get('exit_code', 0), process.stderr
    if 'stderr_contains' in expected:
        assert expected['stderr_contains'] in process.stderr, process.stderr
    if process.returncode:
        return
    if expected['format'] == 'json':
        assert_subset(json.loads(process.stdout), expected['subset'])
        return
    circuit = qasm2.loads(process.stdout, strict=True) if expected['format'] == 'qasm2' else parse(process.stdout)
    assert circuit.num_qubits == expected['qubits'], f'wrong circuit width: {circuit.num_qubits}'
    if 'allowed_gates' in expected:
        unexpected = set(circuit.count_ops()) - set(expected['allowed_gates'])
        assert not unexpected, f'unsupported output gates: {unexpected}'
    if 'unitary' in expected:
        matrix = np.array([[complex(*value) for value in row] for row in expected['unitary']])
        error = np.linalg.norm(Operator(circuit).data - matrix, 2)
    else:
        state = np.zeros(1 << circuit.num_qubits, dtype=complex)
        for value in expected['statevector']:
            state[value['index']] = complex(value['re'], value['im'])
        error = np.linalg.norm(Statevector(circuit).data - state)
    assert error <= expected['tolerance'], f'error {error} exceeds {expected["tolerance"]}'


def main():
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command.pop(0)
    for case in MANIFEST['cases']:
        run_case(case, command or DEFAULT_COMMAND)
        print(case['name'] + ': ok')
    print(f'{len(MANIFEST["cases"])} conformance cases passed')


if __name__ == '__main__':
    main()
