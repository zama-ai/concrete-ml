#!/usr/bin/env bash

set -e

TAG_NAME=""

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
    echo "Please provide a tag name with option '--tag_name'"
    exit 1
fi

# Fetch all remote tags
git fetch --tags --force

# If the tag name is already found remotely, exist with status 1
if git ls-remote --tags origin | grep -q "${TAG_NAME}"; then
    echo "Tag ${TAG_NAME} already exists remotely."
    exit 1
fi
