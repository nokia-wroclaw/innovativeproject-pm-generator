#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/docker_utils.sh"

execute_docker_compose fastapi sh -c "cd /app && uv run mypy app"
