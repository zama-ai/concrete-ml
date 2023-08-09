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

# Check that a version is provided
if [ "$TAG_NAME" == "" ]; then
    echo "Please provide a tag name"
    exit 1
fi

if git tag -l  | grep -q "${TAG_NAME}"; then
    echo "Tag ${TAG_NAME} already exists locally. Should it be deleted ?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) git tag -d "${TAG_NAME}"; break;;
            No ) exit;;
        esac
    done
fi

git fetch --tags --force
if git ls-remote --tags origin | grep -q "${TAG_NAME}"; then
    echo "Tag ${TAG_NAME} already exists remotely. Should it be deleted ?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) git push --delete "${TAG_NAME}"; break;;
            No ) exit;;
        esac
    done
fi
