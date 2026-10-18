#!/usr/bin/env bash
# Run one command in WSL from the repository root with the project's environment.
# From PowerShell: wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Grounded-QA-Service/scripts/wsl_run.sh <command> [args...]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/gqa"
export GQA_DATA="${GQA_DATA:-$HOME/gqa-data}"
export GQA_RUNS="${GQA_RUNS:-$HOME/gqa-runs}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Hugging Face models are pinned in gqa/hf_models.py; never let a load follow upstream main.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PATH="$HOME/.local/bin:$UV_PROJECT_ENVIRONMENT/bin:$PATH"
mkdir -p "$GQA_DATA" "$GQA_RUNS/logs"
exec "$@"
