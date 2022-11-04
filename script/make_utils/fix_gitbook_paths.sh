#!/usr/bin/env bash

set -e

DIRECTORY=$1

FILES=$(find "$DIRECTORY" -name "*.md")

for f in ${FILES}
do
    # sed -i is different on mac and on linux
    sed "s@../developer-guide/api@https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/docs/developer-guide/api@g" "$f" > tmp.file.md
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