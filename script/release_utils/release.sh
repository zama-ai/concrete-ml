#!/usr/bin/env bash

set -e

CML_VERSION="$(poetry version --short)"

CML_VERSION_MAJOR_MINOR=$(echo "$CML_VERSION" | cut -d '.' -f -2)
RELEASE_BRANCH_NAME="release/${CML_VERSION_MAJOR_MINOR}.x"

IS_RC="$(poetry run python ./script/make_utils/version_utils.py isprerelease --version "$CML_VERSION")"

if [[ $(make check_is_on_main_up_to_date) ]]; then
    if ! ${IS_RC}; then
        # Checkout and push release branch to repo
        git checkout -b "${RELEASE_BRANCH_NAME}"
        git lfs fetch --all
        git push public "HEAD:${RELEASE_BRANCH_NAME}"
    fi

    TAG_NAME="v${CML_VERSION}"

    ./script/release_utils/check_tag.sh --tag_name "${TAG_NAME}"

	git fetch --tags --force
	git tag -s -a -m "${TAG_NAME} release" "${TAG_NAME}"
	git push origin "refs/tags/${TAG_NAME}"
    git checkout main

else
    echo "Please start the release process on branch 'main', and put it up to date."
fi
