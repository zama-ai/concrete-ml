#!/bin/bash

set -e

FILES=$(cd docs && find developer-guide/api -name "*.md")

# New apidocs section
TMP_FILE=$(mktemp /tmp/apidocs.XXXXXX)
rm -rf "$TMP_FILE"
touch "$TMP_FILE"

for f in $FILES
do
    filename=$(echo "$f" | rev | cut -d '/' -f 1 | rev)

    echo "  - [$filename]($f)" >> "$TMP_FILE"
done

# Recreate SUMMARY.md
FINAL_FILE="docs/SUMMARY.md"
NEW_FINAL_FILE="docs/SUMMARY.md.tmp"

# Beginning
grep "<!-- auto-created, do not edit, begin -->" $FINAL_FILE -B 100000 > $NEW_FINAL_FILE

# Middle
sort "$TMP_FILE" | grep -v "\[README.md\]" >> $NEW_FINAL_FILE

# End
grep "<!-- auto-created, do not edit, end -->" $FINAL_FILE -A 100000 >> $NEW_FINAL_FILE

# Replace
mv $NEW_FINAL_FILE $FINAL_FILE





