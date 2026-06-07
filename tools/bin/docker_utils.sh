#!/bin/bash

execute_docker_compose() {
    local service="$1"
    shift
    local warning_filter='variable is not set. Defaulting to a blank string.'

    local preferred="${USER:-}-genpm-${service}"
    local container
    container=$(docker ps --format '{{.Names}}' | grep -F "${preferred}" | head -n 1)
    if [ -z "$container" ]; then
        container=$(docker ps --format '{{.Names}}' | grep -E "${service}$" | head -n 1)
    fi

    if [ -n "$container" ]; then
        docker exec -i "$container" "$@" 2> >(grep -Fv "$warning_filter" >&2)
    else
        if command -v docker &> /dev/null && docker compose &> /dev/null; then
            docker compose exec "$service" "$@" 2> >(grep -Fv "$warning_filter" >&2)
        elif command -v docker-compose &> /dev/null; then
            docker-compose exec "$service" "$@" 2> >(grep -Fv "$warning_filter" >&2)
        else
            echo "Błąd: Nie znaleziono kontenera dla usługi: $service" >&2
            exit 1
        fi
    fi
}
