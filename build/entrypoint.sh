#!/bin/bash
set -e

APP_HOME="${APP_HOME:-/home/sparkuser/app}"
export HOME="${APP_HOME}/apps/generator"

if ! mkdir -p "$HOME" 2>/dev/null || [ ! -w "$HOME" ]; then
    export HOME="/tmp"
fi

export JUPYTER_CONFIG_DIR="$HOME/.jupyter"
export JUPYTER_DATA_DIR="$HOME/.local/share/jupyter"
export JUPYTER_RUNTIME_DIR="/tmp/jupyter-runtime"

mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"

start_jupyter() {
    echo "Starting Spark History Server..."
    echo "Starting Jupyter Notebook..."
    $SPARK_HOME/sbin/start-history-server.sh || true
    jupyter notebook --ip=0.0.0.0 --port=4041 --no-browser --NotebookApp.token='' --NotebookApp.password='' --notebook-dir="${APP_HOME}/apps/generator"
}

start_jupyter
