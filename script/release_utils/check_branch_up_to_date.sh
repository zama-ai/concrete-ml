#!/usr/bin/env bash

set -e

git fetch 

# Get latest commits from local and remote brances
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")

# If local and remote commits are not the same, or if some local changes are detected (unpushed 
# changes), the current local branch is not up to date
if [ "$LOCAL" != "$REMOTE" ] || [ -n "$(git status --porcelain)" ]; then
    # Get the current local branch name
    BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)"

    echo "Local branch '$BRANCH_NAME' is not up to date with its remote tracking branch. "
    exit 1
fi
