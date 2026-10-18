#!/usr/bin/env bash
# Create or update the project environment (~/.venvs/gqa) from uv.lock, including the dev group.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/gqa}"
"${UV:-$HOME/.local/bin/uv}" sync --frozen --all-groups
