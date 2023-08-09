#!/usr/bin/env bash

set -e

CHECK_UP_TO_DATE='false'

while [ -n "$1" ]
do
   case "$1" in
        "--up_to_date" )
            CHECK_UP_TO_DATE="true"
            shift
            ;;
   esac
   shift
done

# Get the current default main branch from origin
MAIN_BRANCH="$(git rev-parse --abbrev-ref origin/HEAD | cut -d '/' -f 2)"

# Get the current local branch name
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)"

# Print a message and exit with status code 1 if : 
# - the current local branch is not main
# - the main branch is not up to date and/or has some unpushed files (if 
#   option 'is_on_main_up_to_date' is found)
if [[ "${BRANCH_NAME}" != "${MAIN_BRANCH}" ]]; then											
    echo "Expected branch '${MAIN_BRANCH}', got branch '${BRANCH_NAME}'. Please checkout."
    exit 1
elif ${CHECK_UP_TO_DATE}; then
    git update-index --really-refresh
    if ! (git diff-index --quiet HEAD) && [ -z "$(git status --porcelain)" ]; then
        echo "Please pull the latest changes from branch '${MAIN_BRANCH}' and avoid unpushed files."
        exit 1
    fi
fi
