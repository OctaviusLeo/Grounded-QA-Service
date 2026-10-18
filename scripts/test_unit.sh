#!/usr/bin/env bash
# The single command for the unit + property test suite (Req 0.11):
#   bash scripts/test_unit.sh
# Runs every test not marked model/docker/hf, with statement coverage, then fails if any
# non-trivial gqa/ module is below 80% (scripts/check_coverage.py).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
cov_json="${TMPDIR:-/tmp}/gqa-coverage-$$.json"
"$PY" -m pytest -q -m "not model and not docker and not hf" \
  --cov=gqa --cov-report=term --cov-report="json:$cov_json" "$@"
"$PY" scripts/check_coverage.py "$cov_json"
rm -f "$cov_json"
