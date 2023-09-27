#!/usr/bin/env bash

set -e

# Find all images and md files in the project's tracked directories (i.e., not in .gitignore), 
# including untracked files (i.e., that got created but not staged)
IMAGE_FILES=$(git -C ./docs ls-files "*.png" "*.svg" "*.jpg" "*.jpeg" --cached --others --exclude-standard --full-name)
MD_FILES=$(git -C ./docs ls-files "*.md" --cached --others --exclude-standard --full-name)

# Create an empty list to store image files mentioned in notebooks
MENTIONED_IMG_FILES=()
for img_file in $IMAGE_FILES; do
    # Extract the image file name from the full path
    img_name=$(basename "$img_file")
    # Search if the image is mentioned in any notebooks (".ipynb" files).
    mentioned_in_specific_ipynb=$(find "./docs" -type f -name "*.ipynb" -exec grep -q  -F "$img_name" {} \; -print)
    # Check if the image file extension is valid and if it is mentioned in any notebooks
    if [[ $img_name =~ \.(jpeg|svg|jpg|png)$ ]] && [ -n "$mentioned_in_specific_ipynb" ]; then
      MENTIONED_IMG_FILES+=("$img_file")
    fi
done

# Remove the image file paths mentioned in notebooks from IMAGE_FILES.
for mentioned_img_filename in "${MENTIONED_IMG_FILES[@]}"; do
    IMAGE_FILES=$(echo "$IMAGE_FILES" | grep -vF "$mentioned_img_filename")
done

# shellcheck disable=SC2086
poetry run python script/doc_utils/check_all_images_are_used.py --images $IMAGE_FILES --files $MD_FILES

