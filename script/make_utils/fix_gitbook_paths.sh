#!/usr/bin/env bash

set -e

DIRECTORY=$1

FILES=$(find "$DIRECTORY" -name "*.md")

for f in ${FILES}
do
    # sed -i is different on mac and on linux
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
