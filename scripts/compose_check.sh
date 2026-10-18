#!/usr/bin/env bash
# Integration check for `docker compose up` (Req 9.7): build and start the whole stack detached,
# poll /health until the API accepts requests, send one /answer request, then stop the api
# container (the ollama service keeps running for other jobs).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
docker compose up -d --build
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "health ok after ${i} polls: $(curl -fsS http://127.0.0.1:8000/health)"
    curl -fsS -X POST http://127.0.0.1:8000/answer -H 'content-type: application/json' \
      -d '{"query": "Where was Super Bowl 50 played?"}'
    echo
    curl -fsS http://127.0.0.1:8000/metrics; echo
    exit 0
  fi
  sleep 2
done
echo "FAILED: /health did not accept requests within 360 s"
docker compose ps
docker compose logs --tail 50 api
exit 1
