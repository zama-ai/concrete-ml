#!/usr/bin/env bash

set -e

# Check that the current branch is main and that it is up to date
if ./script/release_utils/check_branch_is_main.sh --up_to_date; then
    exit 1
fi

# Check that all version numbers are coherent and that apidocs are properly built
make check_version_coherence
make check_apidocs

# Get the Concrete ML version to release
CML_VERSION="$(poetry version --short)"

# Release should start from main, up to date
if [[ $(make check_is_on_main_up_to_date) ]]; then

    # Check if this is a release candidate
    IS_RC="$(poetry run python ./script/make_utils/version_utils.py isprerelease --version "$CML_VERSION")"

    # If this is not a release candidate, a release branch is created and pushed to the repository
    if ! ${IS_RC}; then
        # Create the release branch, of the form 'release/(major).(minor).x' 
        CML_VERSION_MAJOR_MINOR=$(echo "$CML_VERSION" | cut -d '.' -f -2)
        RELEASE_BRANCH_NAME="release/${CML_VERSION_MAJOR_MINOR}.x"

        # Checkout and push release branch to repo
        git checkout -b "${RELEASE_BRANCH_NAME}"
        git lfs fetch --all
        git push
    fi
    
    # Get the tag associated to the upcoming release
    TAG_NAME="v${CML_VERSION}"

    # Check that the tag does not exist yet. If it does, ask the user to confirm its removal 
    # (locally and remotely)
    ./script/release_utils/check_tag.sh --tag_name "${TAG_NAME}"

    # Push the tag to the repository, which triggers the release CI
	git fetch --tags --force
	git tag -s -a -m "${TAG_NAME} release" "${TAG_NAME}"
	git push origin "refs/tags/${TAG_NAME}"
    git checkout main

else
    echo "Please start the release process on branch 'main', and put it up to date."
fi
