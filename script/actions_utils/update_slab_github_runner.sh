#!/bin/bash

# This script updates the lab-github-runner in our workflows by replacing the
# lab-github-runner's release commit. 
# Note: For security reasons, we prefer updating the commit instead of the tag.

# Define the release commit to replace
RELEASE_COMMIT="11bd71901bbe5b1630ceea73d27597364c9af683"

for file in $(find .github -type f \( -name "*.yml" -o -name "*.yaml" \)); do
    echo "Processing: $file"

    # Extract lines containing "uses: actions/checkout@"
    echo "Before modifications:"
    grep -nE "uses: actions/checkout@" "$file"

    # Replace the target line with the new commit, any comments will be deleted
    sed -E -i "s|uses: actions/checkout@.*|uses: actions/checkout@$RELEASE_COMMIT|g" "$file"

    # Extract lines containing "uses: actions/checkout@" after modification
    echo "After modifications:"
    grep -nE "uses: actions/checkout@" "$file"
done
