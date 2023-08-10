#!/usr/bin/env bash

CHECK_UP_TO_DATE="false"

while [ -n "$1" ]
do
   case "$1" in
        "--up_to_date" )
            CHECK_UP_TO_DATE="true"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

# Get the current default main branch from origin
MAIN_BRANCH="$(git rev-parse --abbrev-ref origin/HEAD | cut -d '/' -f 2)"

# Get the current local branch name
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)"

# If the current local branch is not main, print a message and exit with status code 1 
if [[ "${BRANCH_NAME}" != "${MAIN_BRANCH}" ]]; then											
    echo "Expected branch '${MAIN_BRANCH}', got branch '${BRANCH_NAME}'. Please checkout."
    exit 1

# Else, if option '--up_to_date' is set to 'true' and main branch is not up to date or has some 
# unpushed local files, print a message and exit with status code 1
elif ${CHECK_UP_TO_DATE}; then
    ./script/release_utils/check_branch_up_to_date.sh
fi
