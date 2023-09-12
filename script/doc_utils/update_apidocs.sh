#!/usr/bin/env bash

set -e

APIDOCS_OUTPUT="$1"

# Clean
rm -rf "$APIDOCS_OUTPUT"

# Ignore concrete.ml.quantization.qat_quantizers since 
# brevitas has some issues with lazydocs
poetry run lazydocs --output-path="$APIDOCS_OUTPUT" \
    --overview-file="README.md" \
    --src-base-url="../../../" \
    --no-watermark \
    --ignored-modules concrete.ml.quantization.qat_quantizers \
    concrete.ml

