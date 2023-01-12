#!/usr/bin/env bash

set -e

DIRECTORY=$1

FILES=$(find "$DIRECTORY" -name "*.md")

# Courtesy of https://stackoverflow.com/questions/6245570/how-do-i-get-the-current-branch-name-in-git
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

for f in ${FILES}
do
    # sed -i is different on mac and on linux
    if [[ "$CURRENT_BRANCH" == "release/"* ]]; then
        FILLER="tree/$CURRENT_BRANCH"
    else
        FILLER="tree/main"
    fi

    sed "s@../developer-guide/api@https://github.com/zama-ai/concrete-ml-internal/$FILLER/docs/developer-guide/api@g" "$f" > tmp.file.md

    mv tmp.file.md "$f"

    # sed -i is different on mac and on linux
    sed "s@\%5C\_@\_@g" "$f" > tmp.file.md
    mv tmp.file.md "$f"
done

FILES=$(find "$DIRECTORY" -name "*.rst")

for f in ${FILES}
do
    # sed -i is different on mac and on linux
    sed "s@\%5C\_@\_@g" "$f" > tmp.file.rst
    mv tmp.file.rst "$f"
done