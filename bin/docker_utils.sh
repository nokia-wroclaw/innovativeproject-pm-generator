#!/bin/bash

execute_docker_compose() {
    warning_filter='variable is not set. Defaulting to a blank string.'

    if command -v docker &> /dev/null && docker compose &> /dev/null; then
        docker compose exec "$@" 2> >(grep -Fv "$warning_filter" >&2)
    elif command -v docker-compose &> /dev/null; then
        docker-compose exec "$@" 2> >(grep -Fv "$warning_filter" >&2)
    else
        exit 1
    fi
}
