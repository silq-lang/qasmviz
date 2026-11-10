#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [[ -d "$VENV_DIR" ]]; then
  echo "Using existing virtual environment at $VENV_DIR"
else
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo "Installing dependencies"
python -m pip install \
  -c "$(dirname "${BASH_SOURCE[0]}")/constraints.txt" \
  'qiskit>=2.5,<3' \
  'qiskit-qasm3-import>=0.6,<0.7' \
  'openqasm3[parser]>=1.0.1,<2' \
  qiskit-ibm-runtime \
  qiskit-aer \
  cirq ply \
  pytket \
  qblaze

echo "Done"
echo "Activate with: source $VENV_DIR/bin/activate"
