#!/usr/bin/env bash

set -e
DIRECTORY=$1
OPTIONS=$2

MD_FILES=$(find "$DIRECTORY" -type f -name "*.md")

# shellcheck disable=SC2086
poetry run python script/doc_utils/fix_double_dollars_issues_with_mdformat.py --files $MD_FILES $OPTIONS
