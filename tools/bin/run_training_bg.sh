#!/bin/bash

set -e

CONTAINER="${USER}-genpm-spark"
WORKDIR="/home/${USER}/app/apps/generator"
PYTHON_FILE=""

usage() {
  echo "Usage: $0 -f <python_file_relative_to_generator_dir>"
  echo ""
  echo "  -f | --file    Python file to run (relative to apps/generator inside container)"
  echo ""
  echo "Example:"
  echo "  $0 -f genpm/modelling/run_training.py"
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -f | --file) PYTHON_FILE="$2"; shift ;;
    -h | --help) usage ;;
    *) echo "Unknown flag: $1" >&2; usage ;;
  esac
  shift
done

if [[ -z "$PYTHON_FILE" ]]; then
  echo "Error: -f <python_file> is required." >&2
  usage
fi

LOG_FILE="/tmp/training_$(date +%Y%m%d_%H%M%S).log"

if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
  echo "Error: container '${CONTAINER}' is not running."
  echo "Start it first with: ./tools/bin/start_env.sh"
  exit 1
fi

if ! docker exec "${CONTAINER}" test -f "${WORKDIR}/${PYTHON_FILE}"; then
  echo "Error: '${PYTHON_FILE}' not found inside container at ${WORKDIR}/${PYTHON_FILE}"
  exit 1
fi

echo "Launching training in '${CONTAINER}' (detached)..."
echo "  File   : ${WORKDIR}/${PYTHON_FILE}"
echo "  Log    : ${LOG_FILE}  (inside container)"

docker exec -d "${CONTAINER}" bash -c "
  cd ${WORKDIR}
  nohup python3 ${PYTHON_FILE} > ${LOG_FILE} 2>&1
"

echo ""
echo "Training is running in the background. Useful commands:"
echo "  Watch logs : docker exec ${CONTAINER} tail -f ${LOG_FILE}"
echo "  Check alive: docker exec ${CONTAINER} pgrep -af python3"
echo "  Kill it    : docker exec ${CONTAINER} pkill -f '${PYTHON_FILE}'"
