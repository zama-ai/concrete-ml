#!/usr/bin/env bash

set -e

USE_PIP_WHEEL='false'
TEST_CODEBLOCKS='false'

while [ -n "$1" ]
do
   case "$1" in
        "--wheel" )
            USE_PIP_WHEEL='true'
            shift
            CONCRETE_NUMPY="$1"
            ;;
        
        "--codeblocks" )
            TEST_CODEBLOCKS='true'
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
python -m pip install poetry==1.5.1 pytest==7.1.1 pandas==1.3.5 tensorflow==2.10.0 tf2onnx==1.13.0 torchvision==0.14.1

# Install additional pytest plugins
python -m pip install pytest-xdist==2.5.0
python -m pip install pytest-randomly==3.12.0
python -m pip install pytest-repeat==0.9.1

if ${USE_PIP_WHEEL}; then
    # Delete the directory where the pypi wheel file will be created (if it already exists)
    rm -rf dist

    # Build the wheel file
    poetry build -f wheel

    # Install the dependencies as PyPI would do using the wheel file, inclusing the current
    # Concrete-Numpy RC version
    PYPI_WHEEL=$(find dist -type f -name "*.whl")
    python -m pip install "${PYPI_WHEEL}"
    python -m pip install "${CONCRETE_NUMPY}"
else
    python -m pip install concrete-ml
fi

# Run our codeblocks or regular tests
if ${TEST_CODEBLOCKS}; then
    # Install additional pytest plugin for codeblock tests 
    python -m pip install pytest-codeblocks==0.14.0

    ./script/make_utils/pytest_codeblocks.sh
else
    poetry run pytest --version
    poetry run pytest --durations=10 -svv \
        --capture=tee-sys \
        -n 4 \
        --randomly-dont-reorganize \
        --randomly-dont-reset-seed 
fi

# Delete the virtual env directory
rm -rf "${PYPI_VENV}"
