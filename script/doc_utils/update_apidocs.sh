#!/usr/bin/env bash

set -e

APIDOCS_OUTPUT="$1"

# Clean
rm -rf "$APIDOCS_OUTPUT"

poetry run lazydocs --output-path="$APIDOCS_OUTPUT" \
    --overview-file="README.md" \
    --src-base-url="../../../" \
    --no-watermark \
    concrete.ml

