#!/bin/bash


if ! docker network inspect shared-network >/dev/null 2>&1; then
  docker network create shared-network
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export AIRFLOW_PROJ_DIR=../apps/airflow/
export ENV_FILE_PATH=../.env
docker compose --file "$SCRIPT_DIR/../../infra/airflow-docker-compose.yml" up -d --build