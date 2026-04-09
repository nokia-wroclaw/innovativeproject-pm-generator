#!/bin/bash

echo "Composing down containers of ${USER}"
docker compose -p "${USER}_project" down