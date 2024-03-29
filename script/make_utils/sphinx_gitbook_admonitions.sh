#!/usr/bin/env bash

set -e
OPTIONS=$1

MD_FILES=$(find docs-copy -type f -name "*.md")

# shellcheck disable=SC2086
poetry run python script/doc_utils/sphinx_gitbook_admonitions.py --files $MD_FILES $OPTIONS
