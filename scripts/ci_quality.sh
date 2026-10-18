#!/usr/bin/env bash
# CI quality job (self-hosted GPU runner): evaluate the documented CI subset with the configuration
# in config/ci.yaml, then run the regression gate against results/baseline.json (Req 10.1-10.3, 10.5).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/gqa-ci"
export PATH="$HOME/.local/bin:$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1
work="${RUNNER_TEMP:-/tmp}/gqa-ci-$$"
mkdir -p "$work"
export GQA_RUNS="$work"

uv sync --frozen --no-dev
docker compose up -d --wait ollama
read -r mode model judge < <(python -c "import yaml; c = yaml.safe_load(open('config/ci.yaml')); print(c['mode'], c['model'], c['judge_model'])")
for m in "$model" "$judge"; do docker compose exec -T ollama ollama pull "$m" > /dev/null; done
python scripts/fetch_hf_models.py
python -m gqa.eval --config config/default.yaml --mode "$mode" --model "$model" --judge-model "$judge" \
  --subset ci --out "$work/ci_subset.json"
python scripts/ci_gate.py --subset-result "$work/ci_subset.json" --baseline results/baseline.json
