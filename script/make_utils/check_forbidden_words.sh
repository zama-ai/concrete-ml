#!/usr/bin/env bash

set -e

MD_FILES=$(find docs -type f -name "*.md")

# shellcheck disable=SC2086
poetry run python script/doc_utils/check_forbidden_words.py --files $MD_FILES


