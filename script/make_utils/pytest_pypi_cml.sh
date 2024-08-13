#!/usr/bin/env bash

set -e

USE_PIP_WHEEL='false'
TEST_CODEBLOCKS='false'
NO_FLAKY='false'
VERSION=''

while [ -n "$1" ]
do
   case "$1" in
        "--wheel" )
            USE_PIP_WHEEL='true'
            ;;
        
        "--codeblocks" )
            TEST_CODEBLOCKS='true'
            ;;

        "--noflaky" )
            NO_FLAKY='true'
            ;;

        "--version" )
            shift
            VERSION="$1"
            ;;   
   esac
   shift
done

# Target virtual env directory, which will be temporary
PYPI_VENV=".pypi_venv"

# Delete the target venv directory (if it already exists)
rm -rf "${PYPI_VENV}"

# Create the virtual env
python -m venv "${PYPI_VENV}"

# Activate the virtual env
# ShellCheck is unable to follow dynamic paths: https://www.shellcheck.net/wiki/SC1091
# shellcheck disable=SC1091,SC1090
source "${PYPI_VENV}/bin/activate"

# Install additional dependencies that are required in order to run our tests but are not included
# in PyPI
# Investigate a better way of managing these dependencies 
python -m pip install --upgrade pip

if ${USE_PIP_WHEEL}; then
    # Delete the directory where the pypi wheel file will be created (if it already exists)
    rm -rf dist
    
    # Install dev dependencies for testing
    poetry install --only dev

    # Build the wheel file
    poetry build -f wheel

    # Install the dependencies as PyPI would do using the wheel file as well as the given
    # Concrete-Python version
    PYPI_WHEEL=$(find dist -type f -name "*.whl")
    python -m pip install --extra-index-url https://pypi.zama.ai/cpu "${PYPI_WHEEL}"

else
    if [ -z "${VERSION}" ]; then
        python -m pip install concrete-ml[dev]
    else
        python -m pip install concrete-ml[dev]=="${VERSION}"
    fi
fi

if ${TEST_CODEBLOCKS}; then
    ./script/make_utils/pytest_codeblocks.sh

# Else, if flaky should not be considered, run 'pytest_no_flaky'
elif ${NO_FLAKY}; then
    make pytest_no_flaky

# Else, run 'pytest_internal_parallel' (instead of `pytest` since we don't want to check for 
# coverage here)
else
    make pytest_internal_parallel
fi

# Delete the virtual env directory
rm -rf "${PYPI_VENV}"
