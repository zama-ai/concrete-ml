#!/usr/bin/env bash

# Check if the given branch name does not refer to an existing remote branch (from origin). If it 
# does, a 1 exit code is returned

set -e

BRANCH_NAME=""

while [ -n "$1" ]
do
   case "$1" in
        "--branch_name" )
            shift
            BRANCH_NAME="$1"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

# Check that a tag name is provided
if [ "$BRANCH_NAME" == "" ]; then
    echo "Please provide a branch name with option '--branch_name'"
    exit 1
fi

# Fetch all remote branches
git fetch --all

# If the branch name is already found remotely, exist with status 1
if [ -n "$(git ls-remote --heads origin "${BRANCH_NAME}")" ]; then
    echo "Branch ${BRANCH_NAME} already exists remotely."
    exit 1
fi
