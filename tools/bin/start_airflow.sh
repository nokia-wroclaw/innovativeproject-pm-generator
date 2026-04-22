#!/bin/bash


if [[ "$(docker network ls -f -p name=shared-network 2> /dev/null)" == "" ]]; then
  docker network create shared-network
fi
export ENV_FILE_PATH=../.env
docker compose --file ../../infra/airflow-docker-compose.yml up -d --build