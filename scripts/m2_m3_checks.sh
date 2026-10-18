#!/usr/bin/env bash
# Re-run the live pipeline check, the M2 gate, the twice-run determinism check, and the M3 gate.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
python scripts/pipeline_check.py
bash scripts/gate_m2.sh
bash scripts/eval_twice.sh
bash scripts/gate_m3.sh
