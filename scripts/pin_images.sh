#!/usr/bin/env bash
set -euo pipefail
for img in debian:bookworm-slim ghcr.io/astral-sh/uv:0.13.1; do bash scripts/docker_pull.sh "$img"; done
