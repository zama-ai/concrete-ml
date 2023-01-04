#!/usr/bin/env bash

set -e

rm -rf .venvpip
python3 -m venv .venvpip

# shellcheck disable=SC1091
source .venvpip/bin/activate

# Update pip and setuptools
pip install -U pip setuptools

# Needed for some tests
pip install pandas
pip install tensorflow
pip install tf2onnx

# Fresh new Concrete-ML from pypi:
# If it is a public version:
#       pip install concrete-ml
# If it is still a private version
pip install -U --pre concrete-ml

# For pytest
pip install pytest
pip install pytest-cov
pip install pytest_codeblocks
pip install pytest-xdist
pip install pytest-randomly
pip install pytest-repeat

./script/make_utils/pytest_codeblocks.sh
