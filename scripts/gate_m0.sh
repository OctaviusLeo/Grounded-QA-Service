#!/usr/bin/env bash
# M0 gate: unit suite + coverage, then the smoke-test evidence checks.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh -q > "${GQA_RUNS:-$HOME/gqa-runs}/logs/gate_unit.log" 2>&1 || { tail -n 30 "${GQA_RUNS:-$HOME/gqa-runs}/logs/gate_unit.log"; echo "FAIL unit suite"; exit 1; }
tail -n 1 "${GQA_RUNS:-$HOME/gqa-runs}/logs/gate_unit.log"
python scripts/gates.py m0
