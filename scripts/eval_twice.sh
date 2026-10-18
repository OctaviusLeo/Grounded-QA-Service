#!/usr/bin/env bash
# Task 4.10: run the harness twice on the CI subset with the same seed, then compare.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
for run in 1 2; do
  python -m gqa.eval --config config/default.yaml --seed 1234 --mode hybrid_rerank --model granite3.3:2b \
    --subset ci --out "results/eval/ci_run$run.json"
done
python scripts/compare_runs.py results/eval/ci_run1.json results/eval/ci_run2.json
