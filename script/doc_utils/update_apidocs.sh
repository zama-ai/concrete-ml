#!/usr/bin/env bash

set -e

APIDOCS_OUTPUT="$1"

# Courtesy of https://stackoverflow.com/questions/6245570/how-do-i-get-the-current-branch-name-in-git
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ "$CURRENT_BRANCH" == "release/"* ]]; then
    # Here, it can be concrete-ml-internal or concrete-ml, depending if files have already
    # been pushed to public repository
    FILLER="concrete-ml-internal/tree/$CURRENT_BRANCH"
else
    FILLER="concrete-ml-internal/tree/main"
fi

# Show which concrete-ml we are going to apidoc
poetry run python -c "import concrete.ml; print(concrete.ml.__version__); print(concrete.ml.__path__)"

poetry run lazydocs --output-path="$APIDOCS_OUTPUT" \
    --overview-file="README.md" \
    --src-base-url="https://github.com/zama-ai/$FILLER/" \
    --no-watermark \
    concrete.ml

