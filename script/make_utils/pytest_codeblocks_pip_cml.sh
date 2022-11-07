#!/usr/bin/env bash

set -e

rm -rf .venvpip
python3.9 -m venv .venvpip

# shellcheck disable=SC1091
source .venvpip/bin/activate

# Needed for some tests
pip install pandas
pip install tensorflow
pip install tf2onnx

# Fresh new Concrete-ML from pypi
pip install concrete-ml

# For pytest
pip install pytest
pip install pytest-cov
pip install pytest_codeblocks
pip install pytest-xdist
pip install pytest-randomly
pip install pytest-repeat

./script/make_utils/pytest_codeblocks.sh
