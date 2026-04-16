#!/bin/bash
set -e

APP_HOME="${APP_HOME:-/home/sparkuser/app}"
export USER="${USER:-sparkuser}"
export LOGNAME="${LOGNAME:-$USER}"
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-$USER}"
export HOME="${APP_HOME}/apps/generator"

if ! mkdir -p "$HOME" 2>/dev/null || [ ! -w "$HOME" ]; then
    export HOME="/tmp"
fi

export JUPYTER_CONFIG_DIR="$HOME/.jupyter"
export JUPYTER_DATA_DIR="$HOME/.local/share/jupyter"
export JUPYTER_RUNTIME_DIR="/tmp/jupyter-runtime"
export SPARK_LOCAL_DIRS="/tmp/spark-local"
export SPARK_WORKER_DIR="/tmp/spark-worker"
export SPARK_LOG_DIR="/tmp/spark-logs"
export IVY_HOME="$HOME/.ivy2"

export SPARK_SUBMIT_OPTS="${SPARK_SUBMIT_OPTS:-} -Duser.name=$USER -Duser.home=$HOME -Divy.home=$IVY_HOME -Divy.cache.dir=$IVY_HOME/cache"

mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" \
    "$SPARK_LOCAL_DIRS" "$SPARK_WORKER_DIR" "$SPARK_LOG_DIR" "$IVY_HOME" "$IVY_HOME/cache" "$IVY_HOME/local"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

start_jupyter() {
    echo "Starting Spark History Server..."
    echo "Starting Jupyter Notebook..."
    $SPARK_HOME/sbin/start-history-server.sh || true
    jupyter notebook --ip=0.0.0.0 --port=4041 --no-browser --NotebookApp.token='' --NotebookApp.password='' --notebook-dir="${APP_HOME}/apps/generator"
}
DEVCONTAINER=false
if [ "$DEVCONTAINER" = "true" ]; then
    echo "lalala"
    rm -rf /home/sparkuser/app/.venv
    uv sync --package genpm-generator --no-cache
    source /home/sparkuser/app/.venv/bin/activate
    uv run python -m ipykernel install --user --name=spark-env --display-name "Python (Spark Project)"

fi

start_jupyter
