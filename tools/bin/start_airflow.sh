#!/bin/bash
set -euo pipefail

if ! docker network inspect shared-network >/dev/null 2>&1; then
  docker network create shared-network
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/infra/airflow-docker-compose.yml"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-${USER:-local}-genpm}"

export USER="${USER:-$(id -un)}"
export AIRFLOW_PROJ_DIR="$REPO_ROOT/apps/airflow"
export ENV_FILE_PATH="$REPO_ROOT/.env"
export AIRFLOW_UID="${AIRFLOW_UID:-50000}"

if [ ! -f "$REPO_ROOT/.env" ]; then
  echo "Missing $REPO_ROOT/.env — run tools/bin/start_env.sh or copy .env.template"
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$REPO_ROOT/.env"
set +a

if [ -z "${FERNET_KEY:-}" ]; then
  export FERNET_KEY="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
  echo "FERNET_KEY=$FERNET_KEY" >> "$REPO_ROOT/.env"
  echo "Appended FERNET_KEY to .env"
fi

echo "Airflow compose project: $PROJECT_NAME"
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" up -d --build

SPARK_CONTAINER="${USER}-genpm-spark"
if ! docker ps --format '{{.Names}}' | grep -qx "$SPARK_CONTAINER"; then
  echo "WARNING: $SPARK_CONTAINER is not running. Visualization DAG needs Spark — run: tools/bin/start_env.sh"
fi
