#!/usr/bin/env bash
# Pull one image and print its repo digest. Usage: bash scripts/docker_pull.sh <image:tag>
set -euo pipefail
docker pull --quiet "$1"
docker image inspect --format '{{index .RepoDigests 0}}' "$1"
