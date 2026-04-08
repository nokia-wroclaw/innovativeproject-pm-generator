#!/bin/bash
set -e

sudo chown -R sparkuser:sparkuser $SPARK_HOME/logs $SPARK_HOME/event_logs

setup_env() {
    echo "Looking for pyproject.toml in $(pwd)"

    if [ -f "pyproject.toml" ]; then
        echo "pyproject.toml found creating .venv"
        uv sync

        uv run python -m ipykernel install --user --name=spark-env --display-name "Python (Spark Project)"
    else
        echo "Couldn' t find any pyproject.toml in $(pwd)"
    fi
}

start_jupyter() {
    echo "Starting Spark History Server..."
    echo "Starting Jupyter Notebook..."
    $SPARK_HOME/sbin/start-history-server.sh && jupyter notebook --ip=0.0.0.0 --port=4041 --no-browser --NotebookApp.token='' --NotebookApp.password=''
}

setup_env
start_jupyter