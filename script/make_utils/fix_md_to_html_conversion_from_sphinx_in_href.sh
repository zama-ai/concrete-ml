#!/usr/bin/env bash

set -e
DIRECTORY=$1
OPTIONS=$2

MD_FILES=$(find "$DIRECTORY" -type f -name "*.md")

# Replace `href="*.md` patterns with `href="*.html` because Sphinx does not handle these
# shellcheck disable=SC2086
poetry run python script/doc_utils/fix_md_to_html_conversion_from_sphinx_in_href.py --files $MD_FILES
