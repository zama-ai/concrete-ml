#!/usr/bin/env bash

set -e

UTILS_FILES=$(find use_case_examples -type f -name "utils.py")

if [ -n "$UTILS_FILES" ]; then \
    echo -n -e "Files should not be named 'utils.py' in use case directories as this can lead "; \
    echo "to import issues. Please rename the following files:"; \
    echo "$UTILS_FILES"
    echo
    exit 1; \
fi
