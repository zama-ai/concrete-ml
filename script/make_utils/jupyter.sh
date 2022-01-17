#!/usr/bin/env bash

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

if isDockerContainer; then
    poetry run jupyter notebook --allow-root --no-browser --ip=0.0.0.0
else
    poetry run jupyter notebook --allow-root
fi

