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
  qiskit \
  qiskit-qasm3-import \
  qiskit-ibm-runtime \
  qiskit-aer \
  qblaze

echo "Done"
echo "Activate with: source $VENV_DIR/bin/activate"
