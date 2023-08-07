#!/usr/bin/env bash

set -e

# Get the current branch name
MAIN_BRANCH="$(git rev-parse --abbrev-ref origin/HEAD | cut -d '/' -f 2)"
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)"

PROJECT_VER="$(poetry version --short)"

CML_VERSION_MAJOR_MINOR=$(echo "$CML_VERSION" | cut -d '.' -f -2)
EXPECTED_BRANCH_NAME="release/${CML_VERSION_MAJOR_MINOR}.x"

if [[ "${BRANCH_NAME}" == "$EXPECTED_BRANCH_NAME" ]]; then	
    echo
elif [[ "${BRANCH_NAME}" != "${MAIN_BRANCH}" ]]; then											
    echo "Please checkout to the main branch and pull the latest changes before starting to release."
    echo "Expected branch '${MAIN_BRANCH}', got branch '${BRANCH_NAME}'."
    exit 1
else
    git update-index --really-refresh
    if !(git diff-index --quiet HEAD); then
        echo "Please pull the latest changes and avoid staged or untracked files before starting to release."
    else
        echo
    fi
fi
