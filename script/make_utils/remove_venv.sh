#!/usr/bin/env bash

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

VENV_PATH=

if isDocker; then
    VENV_PATH="${HOME}/dev_venv/"
else
    VENV_PATH=.venv/
fi

rm -rf "${VENV_PATH}"*
echo "${VENV_PATH}"
