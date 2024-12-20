#!/bin/bash

# This script updates the lab-github-runner in our workflows by replacing the
# lab-github-runner's release commit. 
# Note: For security reasons, we prefer updating the commit instead of the tag.

# Define the release commit to replace
RELEASE_COMMIT="79939325c3c429837c10d6041e4fd8589d328bac"  # v1.4.1

find .github -type f \( -name "*.yml" -o -name "*.yaml" \) -print0 | while IFS= read -r -d '' file; do

    # Check if the file contains the target string
    if grep -q "uses: zama-ai/slab-github-runner@" "$file"; then
        echo "Processing: $file"

        # Extract lines containing "uses: zama-ai/slab-github-runner@" before modification
        echo "Before modifications:"
        grep -nE "uses: zama-ai/slab-github-runner@" "$file"

        # Replace the target line with the new commit; any comments will be deleted
        sed -E -i "s|uses: zama-ai/slab-github-runner@.*|uses: zama-ai/slab-github-runner@$RELEASE_COMMIT|g" "$file"

        # Extract lines containing "uses: zama-ai/slab-github-runner@" after modification
        echo "After modifications:"
        grep -nE "uses: zama-ai/slab-github-runner@" "$file"

        echo "-------------------------"
    fi
done