#!/usr/bin/env bash
set -euo pipefail

# Create a reproducible demo environment without modifying the user's base Python.
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-3.11}"  # a version: uv finds or downloads it

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv venv --clear "${ENV_DIR}" --python "${PYTHON_BIN}"  # rerun to rebuild from scratch
uv pip install --python "${ENV_DIR}/bin/python" -r requirements.txt

echo
echo "Environment ready. Activate it with:"
echo "  source ${ENV_DIR}/bin/activate"
