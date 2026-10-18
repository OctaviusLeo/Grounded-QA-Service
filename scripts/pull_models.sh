#!/usr/bin/env bash
# Start the Model_Server and pull the local models. Usage: bash scripts/pull_models.sh [model ...]
# Default models: granite3.3:2b granite3.3:8b. Prints each model's full manifest digest.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
models=("$@"); [ "${#models[@]}" -eq 0 ] && models=(granite3.3:2b granite3.3:8b)
docker compose up -d --wait ollama
for m in "${models[@]}"; do
  echo "pulling $m"
  docker compose exec -T ollama ollama pull "$m" 2>&1 | tail -n 2
done
curl -fsS http://127.0.0.1:8090/api/tags | python3 -c 'import json,sys; [print(m["name"], m["digest"], m["details"].get("quantization_level"), m["size"]) for m in json.load(sys.stdin)["models"]]'
