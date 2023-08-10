#!/usr/bin/env bash

set -e

TAG_NAME=''

while [ -n "$1" ]
do
   case "$1" in
        "--tag_name" )
            shift
            TAG_NAME="$1"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

# Check that a tag name is provided
if [ "$TAG_NAME" == "" ]; then
    echo "Please provide a tag name"
    exit 1
fi

# Running this script expects the current tag name to be pushed with the current commit. If a tag
# with the same name is found, this means the previous release atempt failed and therefore all 
# associated should be removed

# If the tag name is already found locally, it should be removed
if git tag -l  | grep -q "${TAG_NAME}"; then
    echo "Tag ${TAG_NAME} already exists locally. Should it be deleted ?"

    # Ask the user to confirm the tag's local removal
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) git tag -d "${TAG_NAME}"; break;;
            No ) exit;;
        esac
    done
fi

# Fetch all remote tags
git fetch --tags --force

# If the tag name is already found remotely, it should be removed
if git ls-remote --tags origin | grep -q "${TAG_NAME}"; then
    echo "Tag ${TAG_NAME} already exists remotely. Should it be deleted ?"

    # Ask the user to confirm the tag's remote removal
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) git push --delete "${TAG_NAME}"; break;;
            No ) exit;;
        esac
    done
fi
