#!/usr/bin/env bash
# Integration check for `docker compose up` (Req 9.7): build and start the whole stack detached,
# poll /health until the API accepts requests, then send one answerable and one unanswerable
# question. Writes the responses to results/compose_check.txt. The stack is left running.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
out=results/compose_check.txt
docker compose up -d --build
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health > /dev/null 2>&1; then
    {
      echo "# docker compose up -d --build, then /health accepted after ${i} polls (2 s apart)"
      echo "health: $(curl -fsS http://127.0.0.1:8000/health)"
      for q in "Who was Edward the Confessor's half-brother?" "Who won the 2030 FIFA World Cup?"; do
        body=$(python3 -c 'import json, sys; print(json.dumps({"query": sys.argv[1]}))' "$q")
        echo "POST /answer $body -> $(curl -fsS -X POST http://127.0.0.1:8000/answer -H 'content-type: application/json' -d "$body")"
      done
      echo "metrics: $(curl -fsS http://127.0.0.1:8000/metrics)"
      echo "services:"
      docker compose ps --format '{{.Service}} {{.State}} {{.Status}}'
    } | tee "$out"
    exit 0
  fi
  sleep 2
done
echo "FAILED: /health did not accept requests within 360 s"
docker compose ps
docker compose logs --tail 50 api
exit 1
