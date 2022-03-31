#!/usr/bin/env bash

CURR_DIR=$(dirname "$0")
DOCKER_BUILDKIT=1 docker build --pull --no-cache -f "$CURR_DIR/Dockerfile.release" \
--secret id=build-env,src="${1}" \
-t concrete-ml-release "$CURR_DIR/.."
