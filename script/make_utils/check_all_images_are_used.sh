#!/usr/bin/env bash

set -e

# Find all images and md files in the project's tracked directories (i.e., not in .gitignore), 
# including untracked files (i.e., that got created but not staged)
IMAGE_FILES=$(git -C ./docs ls-files "*.png" "*.svg" "*.jpg" "*.jpeg" --cached --others --exclude-standard --full-name)
MD_FILES=$(git -C ./docs ls-files "*.md" --cached --others --exclude-standard --full-name)

# shellcheck disable=SC2086
poetry run python script/doc_utils/check_all_images_are_used.py --images $IMAGE_FILES --files $MD_FILES

