#!/usr/bin/env bash

set -e

# Get the current branch name
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

CML_VERSION=$1

if [[ -n "$CML_VERSION" ]]; then
    CML_VERSION_START=$( echo "$CML_VERSION" | cut -d '.' -f -2 )
    EXPECTED_BRANCH_NAME="release/${CML_VERSION_START}.x"

    # Given a version a.b.c, the current release branch name is expected to have 
    # format: release/a.b.x
    if [[ "$$BRANCH_NAME" != "$EXPECTED_BRANCH_NAME" ]]; then											
        echo "The current branch name does not match the expected release branch name format."
        echo "Expected $EXPECTED_BRANCH_NAME, but got $BRANCH_NAME."
        exit 1
    fi
else
    echo "Please provide the project's curent version as input argument."
fi
