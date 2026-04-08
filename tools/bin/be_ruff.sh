#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/docker_utils.sh"

if [[ "${1:-}" == "--fix" ]]; then
	execute_docker_compose app sh -c "cd /app && uv run ruff check --fix app"
else
	execute_docker_compose app sh -c "cd /app && uv run ruff check app"
fi
