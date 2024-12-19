#!/bin/bash

# This script updates the lab-github-runner in our workflows by replacing the
# lab-github-runner's release commit. 
# Note: For security reasons, we prefer updating the commit instead of the tag.

# Define the release commit to replace
RELEASE_COMMIT="f26b8d611b2e695158fb0a6980834f0612f65ef8"

echo "Current directory: $(pwd)"

# Search for files and process them
for file in $(find .github -type f \( -name "*.yml" -o -name "*.yaml" \)); do
    echo "Processing: $file"

    # Extract lines containing "uses: actions/checkout@"
    echo "Before modifications:"
    grep -nE "uses: actions/checkout@" "$file" || echo "No match found"

    # Replace the target line with the new commit, keeping indentation intact
    sed -E -i "s|(uses: actions/checkout@)[^[:space:]]*|\1$RELEASE_COMMIT|g" "$file"

    # Extract lines containing "uses: actions/checkout@" after modification
    echo "After modifications:"
    grep -nE "uses: actions/checkout@" "$file" || echo "No match found"

    echo "Updated: $file"
    echo "-------------------------"
done
