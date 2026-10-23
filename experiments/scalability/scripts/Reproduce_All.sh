#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GARQ_PYTHON:-python}"
exec "$PY" "$ROOT/run_scalability.py" "$@" --stage all
