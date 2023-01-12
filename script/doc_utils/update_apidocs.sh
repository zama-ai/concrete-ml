#!/usr/bin/env bash

set -e

APIDOCS_OUTPUT="./docs/developer-guide/api"

# Courtesy of https://stackoverflow.com/questions/6245570/how-do-i-get-the-current-branch-name-in-git
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ "$CURRENT_BRANCH" == "release/"* ]]; then
    # Here, it can be concrete-ml-internal or concrete-ml, depending if files have already
    # been pushed to public repository
    FILLER="concrete-ml-internal/tree/$CURRENT_BRANCH"
else
    FILLER="concrete-ml-internal/tree/main"
fi

poetry run lazydocs --output-path="$APIDOCS_OUTPUT" \
    --overview-file="README.md" \
    --src-base-url="https://github.com/zama-ai/$FILLER/" \
    --no-watermark \
    concrete.ml

