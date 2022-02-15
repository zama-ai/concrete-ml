#!/bin/bash

# Run benchmarks while logging the intermediate results
# Publish findings in the progress tracker

set -e

DEV_VENV_PATH="/home/dev_user/dev_venv"

# Create the file if it does not exist to avoid crashing
touch .env
# shellcheck disable=SC1090,SC1091
source .env
# Don't keep the file, once it's sourced it served its purpose
rm -rf .env

# shellcheck disable=SC1090,SC1091
if ! source "${DEV_VENV_PATH}/bin/activate"; then
    python3 -m venv "${DEV_VENV_PATH}"
    # shellcheck disable=SC1090,SC1091
    source "${DEV_VENV_PATH}/bin/activate"
fi

cd /src/ && make setup_env

mkdir -p /tmp/keycache
mkdir -p logs

ml_log=logs/latest.ml.log
rm -f "${ml_log}"
make -s benchmark 2>&1 | tee -a "${ml_log}"

curl \
-H "Authorization: Bearer ${ML_PROGRESS_TRACKER_TOKEN}" \
-H "Content-Type: application/json" \
-d @progress.json \
-X POST "${ML_PROGRESS_TRACKER_URL}"/measurement
