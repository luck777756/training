#!/usr/bin/env bash
set -euo pipefail

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python
fi

$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY train_model.py
