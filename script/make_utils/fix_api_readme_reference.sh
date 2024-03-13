#!/usr/bin/env bash

set -e
DIRECTORY=$1
OPTIONS=$2

MD_FILES=$(find "$DIRECTORY" -type f -name "index.rst")

# Replace `references/api/README.md` with `_apidoc/modules.html`.
# shellcheck disable=SC2086
poetry run python script/doc_utils/fix_api_readme_reference.py --files $MD_FILES
