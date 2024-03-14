#!/usr/bin/env bash

# Find any isolated $ signs in the docs, except developer guide where 
# some codeblocks will show bash examples
# Only double dollar signs $$ should be used in order to 
# show properly in gitbook

# We check for 3 patterns:
# - line or word starting with a $ not followed by a $
# - a $ enclosed by two characters that are not not $
# - a line that ends with a $ which is not preceeded by a $

# MacOS's grep is different from GNU's so we need to differenciate here.
if [[ $(uname) == "Darwin" ]]; then
  OUT=$(find docs -name "*.md" -not -path "docs/developer/*" -print0 | xargs -0 grep -n -E '((^| |^$)\$[^$])|([^$]\$$)|([^$]$[^$])')
else
  OUT=$(find docs -name "*.md" -not -path "docs/developer/*" -print0 | xargs -0 grep -n -P '((^| |^$)\$[^$])|([^$]\$$)|([^$]$[^$])')
fi

if [ -n "${OUT}" ]; then
  echo "${OUT}"
  exit 1
else
  exit 0
fi
