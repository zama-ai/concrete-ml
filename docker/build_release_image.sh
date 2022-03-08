#!/usr/bin/env bash

CURR_DIR=$(dirname "$0")
DOCKER_BUILDKIT=1 docker build --pull --no-cache -f "$CURR_DIR/Dockerfile.release" \
-t concrete-ml-release "$CURR_DIR/.."
