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
            shift
            CONCRETE_PYTHON="$1"
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
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2685
python -m pip install --upgrade pip
python -m pip install pytest==7.4.1 pandas==2.0.3 tensorflow==2.12.0 tf2onnx==1.15.0 torchvision==0.14.1

# Install additional pytest plugins
python -m pip install pytest-xdist==3.3.1
python -m pip install pytest-randomly==3.15.0
python -m pip install pytest-repeat==0.9.1

if ${USE_PIP_WHEEL}; then
    # Delete the directory where the pypi wheel file will be created (if it already exists)
    rm -rf dist

    # Build the wheel file
    poetry build -f wheel

    # Install the dependencies as PyPI would do using the wheel file as well as the given
    # Concrete-Python version
    PYPI_WHEEL=$(find dist -type f -name "*.whl")
    python -m pip install "${PYPI_WHEEL}"
    python -m pip install "${CONCRETE_PYTHON}"
else
    if [ -z "${VERSION}" ]; then
        python -m pip install concrete-ml
    else
        python -m pip install concrete-ml=="${VERSION}"
    fi
fi

# If codeblocks are checked, install the pytest codeblock plugin first
if ${TEST_CODEBLOCKS}; then
    python -m pip install pytest-codeblocks==0.14.0

    ./script/make_utils/pytest_codeblocks.sh

# Else, if flaky should not be considered, run 'pytest_no_flaky'
elif ${NO_FLAKY}; then
    make pytest_no_flaky

# Else, intall the pytest coverage plugin and run 'pytest_internal_parallel' (instead of `pytest`
# since we don't want to check for coverage here)
else
    python -m pip install pytest-cov==4.1.0
    make pytest_internal_parallel
fi

# Delete the virtual env directory
rm -rf "${PYPI_VENV}"
