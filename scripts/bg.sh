#!/usr/bin/env bash
# Start a long job detached so it survives the calling terminal.
# Usage: bash scripts/bg.sh <name> <command> [args...]
# Writes $GQA_RUNS/logs/<name>.log, <name>.pid and, when the job ends, <name>.exit (its exit status).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
name="$1"; shift
d="${GQA_RUNS:-$HOME/gqa-runs}/logs"
mkdir -p "$d"
rm -f "$d/$name.exit" "$d/$name.pid"
setsid nohup bash -c 'n="$1"; d="$2"; shift 2; echo "$$" > "$d/$n.pid"; bash scripts/wsl_run.sh "$@"; echo "$?" > "$d/$n.exit"' \
  _ "$name" "$d" "$@" > "$d/$name.log" 2>&1 < /dev/null &
for _ in $(seq 50); do [ -s "$d/$name.pid" ] && break; sleep 0.1; done
echo "started $name (pid $(cat "$d/$name.pid" 2>/dev/null || echo unknown)); log: $d/$name.log"
