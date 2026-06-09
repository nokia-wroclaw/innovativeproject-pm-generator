#!/bin/bash

set -e

CREATE_BUCKET=false
BUILD_MODE="auto"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -q | --quiet) echo "milczenie jest zlotem ;)" ;;
    -b | --build) BUILD_MODE="build" ;;
    -n | --no-build) BUILD_MODE="no-build" ;;
    -cb | --create-bucket) CREATE_BUCKET=true ;;
    *) echo "unknown flag: $1" >&2 ; exit 1;
  esac
  shift
done


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="$REPO_ROOT/build"

cd "$REPO_ROOT"

if [ ! -f .env ]; then
  if [ -f .env.template ]; then
    cp .env.template .env
    echo "copied .env.template to .env"
    echo "" >> .env
    echo "USER_GID=$(id -g)" >> .env
    echo "USER_UID=$(id -u)" >> .env
    echo "USER=$USER" >> .env
  else
    echo "Error, couldnt create .env, .env.template is missing in $REPO_ROOT"
    exit 1
  fi

  SYS_IP=$(grep "VM_PUBLIC_IP=" /etc/environment | cut -d'=' -f2 | tr -d '"')
  SYS_PORT=$(grep "VM_SSH_PORT=" /etc/environment | cut -d'=' -f2 | tr -d '"')


  if [ -n "$SYS_IP" ] && [ -n "$SYS_PORT" ]; then
    echo -e "\nVM_PUBLIC_IP=$SYS_IP" >> .env
    echo "VM_SSH_PORT=$SYS_PORT" >> .env
  else
    echo "error: Nie znaleziono VM_PUBLIC_IP lub VM_SSH_PORT w /etc/environment!"
    exit 1
  fi
fi

set -a
[ -f .env ] && source .env
set +a

: "${VM_PUBLIC_IP?Błąd: VM_PUBLIC_IP nie jest ustawiony}"
: "${VM_SSH_PORT?Błąd: VM_SSH_PORT nie jest ustawiony}"

case "$USER" in
  "macsko6154")
    export offset=0
    ;;
  "filant8886")
    export offset=1
    ;;
  "miklep2163")
    export offset=2
    ;;
  "paojar7185")
    export offset=3
    ;;
  "grzwia6937")
    export offset=4
    ;;
  *)
    echo "unknown user, falling back to default ports"
    export offset=5
    ;;
esac

PORT_OFFSET=$((offset * 100))

export JUPYTER_PORT=$((4041 + PORT_OFFSET))
export SPARK_UI_PORT=$((4040 + PORT_OFFSET))
export SPARK_MASTER_PORT=$((18080 + PORT_OFFSET))
export FASTAPI_PORT=$((8000 + PORT_OFFSET))
export FRONTEND_PORT=$((5173 + PORT_OFFSET))
export POSTGRES_PORT=$((5432 + PORT_OFFSET))
export S3_API_PORT=$((9000 + PORT_OFFSET))
export S3_WEBCONSOLE_PORT=$((9001 + PORT_OFFSET))
export KEYCLOAK_PORT=$((8080 + PORT_OFFSET))
export DEVCONTAINER_SSH_PORT=$((2222 + PORT_OFFSET))
export SPARK_MASTER=$((7077 + PORT_OFFSET))
export FRONTEND_ORIGIN="http://localhost:${FRONTEND_PORT}"
export GENPM_SPARK_EXECUTOR_PYTHON="${GENPM_SPARK_EXECUTOR_PYTHON:-/home/${USER}/app/.venv/bin/python3}"


# Write devcontainer env vars to .bashrc (run only once and reenter VS code process)
grep -q "^export COMPOSE_PROJECT_NAME=" ~/.bashrc || echo 'export COMPOSE_PROJECT_NAME="${USER}-genpm"' >> ~/.bashrc
grep -q "^export DEVCONTAINER_SSH_PORT=" ~/.bashrc || echo "export DEVCONTAINER_SSH_PORT=$DEVCONTAINER_SSH_PORT" >> ~/.bashrc
grep -q "^export USER_UID=" ~/.bashrc            || echo 'export USER_UID=$(id -u)' >> ~/.bashrc
grep -q "^export USER_GID=" ~/.bashrc            || echo 'export USER_GID=$(id -g)' >> ~/.bashrc
grep -q "^export USER=" ~/.bashrc                || echo "export USER=$USER" >> ~/.bashrc

echo "Starting environment for $USER-$USER_UID on ports: Frontend=$FRONTEND_PORT, Jupyter=$JUPYTER_PORT, SparkUI=$SPARK_UI_PORT, SparkMaster=$SPARK_MASTER_PORT, MinIO_API=$S3_API_PORT, MinIO_UI=$MINIO_WEBCONSOLE_PORT, FastAPI=$FASTAPI_PORT, Keycloak=$KEYCLOAK_PORT, Postgres=$POSTGRES_PORT"

if [ -d .venv ]; then
  venv_python=".venv/bin/python3"
  venv_ok=true
  if [ ! -e "$venv_python" ]; then
    venv_ok=false
  elif ! readlink -f "$venv_python" >/dev/null 2>&1; then
    venv_ok=false
  elif ! "$venv_python" -c "import sys" >/dev/null 2>&1; then
    venv_ok=false
  fi
  if [ "$venv_ok" = false ]; then
    echo "Removing unusable .venv (likely copied from another account); uv will recreate it."
    rm -rf .venv
  fi
fi

uv sync --quiet

if command -v nvidia-smi &> /dev/null; then
  if ! command -v nvidia-ctk &> /dev/null; then
    echo "There is nvdida card installed"

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit jq

    echo "Configuring Dockera for NVIDIA runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker

    if [ -f /etc/docker/daemon.json ] && ! grep -q '"nvidia"' /etc/docker/daemon.json; then
        echo "confilict in daemon.json"
        cat /etc/docker/daemon.json | jq '.runtimes.nvidia = {"path": "nvidia-container-runtime", "runtimeArgs": []}' | sudo tee /etc/docker/daemon.json > /dev/null
    fi

    sudo systemctl daemon-reload
    sudo systemctl restart docker
    echo "Done."
  else
    echo "NVIDIA Container Toolkit is already installed."
  fi
else
  echo "No nvidia drivers."
fi

if [ "$BUILD_MODE" = "auto" ]; then
  if docker image inspect genpm/spark-jupyter:latest >/dev/null 2>&1 && docker image inspect genpm/fastapi:latest >/dev/null 2>&1; then
    BUILD_MODE="no-build"
  else
    BUILD_MODE="build"
  fi
fi

if ! docker network inspect shared-network >/dev/null 2>&1; then
  docker network create shared-network
fi

if [ "$BUILD_MODE" = "build" ]; then
  docker compose -p "${USER}_project" -f infra/docker-compose.yml up -d --build
else
  docker compose -p "${USER}_project" -f infra/docker-compose.yml up -d
fi

if [ "$CREATE_BUCKET" = true ]; then
  docker compose -p "${USER}_project" -f infra/docker-compose.yml exec -T minio mc alias set myminio http://localhost:9000 ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY}
  docker compose -p "${USER}_project" -f infra/docker-compose.yml exec -T minio mc mb myminio/${S3_BUCKET} --ignore-existing
  docker compose -p "${USER}_project" -f infra/docker-compose.yml exec -T minio mc anonymous set public myminio/${S3_BUCKET}

fi


echo "======================================================================"
echo "Use this locally to route docker ports through OUR port on vm"
echo ""
echo "ssh -L $SPARK_MASTER:localhost:$SPARK_MASTER -L 9005:localhost:9005 -L $FRONTEND_PORT:localhost:$FRONTEND_PORT -L $JUPYTER_PORT:localhost:$JUPYTER_PORT -L $SPARK_MASTER_PORT:localhost:$SPARK_MASTER_PORT -L $SPARK_UI_PORT:localhost:$SPARK_UI_PORT -L $S3_API_PORT:localhost:$S3_API_PORT -L $S3_WEBCONSOLE_PORT:localhost:$S3_WEBCONSOLE_PORT -L $FASTAPI_PORT:localhost:$FASTAPI_PORT -L $KEYCLOAK_PORT:localhost:$KEYCLOAK_PORT -p $VM_SSH_PORT $USER@$VM_PUBLIC_IP"
echo ""
echo "Port Mapping:"
echo "9005 - Airflow"
echo "$FRONTEND_PORT - Frontend"
echo "$JUPYTER_PORT - Jupyter Notebook"
echo "$SPARK_MASTER_PORT - Spark Master"
echo "$SPARK_UI_PORT - Spark UI"
echo "$SPARK_MASTER" - Spark master other port
echo "$S3_API_PORT - S3 API (MinIO)"
echo "$S3_WEBCONSOLE_PORT - S3 Console (MinIO)"
echo "$FASTAPI_PORT - FastAPI Backend"
echo "$KEYCLOAK_PORT - Keycloak"
echo "======================================================================"
