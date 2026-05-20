#!/bin/bash

DEL_VOLUMES=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -v | --volumes) DEL_VOLUMES=true  ;;
    *) echo "unknown flag: $1" >&2 ; exit 1;
  esac
  shift
done

echo "Composing down containers of ${USER}"

if $DEL_VOLUMES; then
  docker compose -p "${USER}_project" down -v
else
  docker compose -p "${USER}_project" down
fi
