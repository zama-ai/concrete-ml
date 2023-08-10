#!/usr/bin/env bash

set -e

CML_VERSION=''
OPEN_PR='false'

while [ -n "$1" ]
do
   case "$1" in
        "--version" )
            shift
            CML_VERSION="$1"
            ;;
        
        "--open_pr" )
            OPEN_PR="true"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

# Check that a version is provided
if [ "$CML_VERSION" == "" ]; then
    echo "Please provide a Concrete ML version"
    exit 1
fi

# Check that the current branch is main and that it is up to date
./script/release_utils/check_branch_is_main.sh --up_to_date

# Update the current version and build apidocs
make set_version
make apidocs

# If the current working directory is not up to date (changes to be pulled, unpushed files), this 
# means version and/or apidocs were updated and thus need to be pushed
# The following assumes that the current branch is main and that the given version is right
if ! ./script/release_utils/check_branch_up_to_date.sh; then

    BRANCH_NAME="chore/prepare_release_${CML_VERSION}"
    COMMIT_MESSAGE="chore: prepare release ${CML_VERSION}"
    
    # Checkout, commit and push changes
    git checkout -b "${BRANCH_NAME}"
    git add .
    git commit -m "${COMMIT_MESSAGE}"
    git push

    # Open a PR with GitHub command line if enabled
    if ${OPEN_PR}; then
        gh auth login
        gh pr create --title "${COMMIT_MESSAGE}" --body "Preparation for release ${CML_VERSION}"
    
    # Else, ask the user to open the PR on GitHub
    else
        echo "Please open a pull-request on github for branch '${BRANCH_NAME}', targeting branch 'main'."
        git checkout main
    fi

# If the working direcotry is up to date, nothing needs to be pushed and the release can be started 
else
    echo "Nothing to push, version and apidocs are up to date. Ready to release."
fi
