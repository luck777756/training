#!/usr/bin/env bash
set -euo pipefail

for exe in python3 python py; do
  if command -v "$exe" >/dev/null 2>&1; then
    PY="$exe"
    break
  fi
done

if [ -z "${PY-}" ]; then
  echo "Error: python not found" >&2
  exit 1
fi

$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY train_model.py