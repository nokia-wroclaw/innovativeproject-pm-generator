#!/bin/bash

set -e

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
    echo "Error, couldnt create .env, .env.template is missing in $BUILD_DIR"
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
export POSTGRES_PORT=$((5432 + PORT_OFFSET))
export MINIO_API_PORT=$((9000 + PORT_OFFSET))
export MINIO_WEBCONSOLE_PORT=$((9001 + PORT_OFFSET))


echo "Starting environment for $USER-$USER_UID on ports: Jupyter=$JUPYTER_PORT, SparkUI=$SPARK_UI_PORT, SparkMaster=$SPARK_MASTER_PORT, MinIO_API=$MINIO_API_PORT, MinIO_UI=$MINIO_WEBCONSOLE_PORT, FastAPI=$FASTAPI_PORT, Postgres=$POSTGRES_PORT"

docker compose -f infra/docker-compose.yml up -d --build


echo "======================================================================"
echo "Use this locally to route a docker ports through OUR port on vm"
echo ""
echo "ssh -L $JUPYTER_PORT:localhost:$JUPYTER_PORT -L $SPARK_MASTER_PORT:localhost:$SPARK_MASTER_PORT -L $SPARK_UI_PORT:localhost:$SPARK_UI_PORT -L $MINIO_API_PORT:localhost:$MINIO_API_PORT -L $MINIO_WEBCONSOLE_PORT:localhost:$MINIO_WEBCONSOLE_PORT -L $FASTAPI_PORT:localhost:$FASTAPI_PORT -p $VM_SSH_PORT $USER@$VM_PUBLIC_IP"
echo ""
echo "======================================================================"