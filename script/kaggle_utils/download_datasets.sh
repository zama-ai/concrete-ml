#!/bin/bash

set -e

# You need to have a valid ~/.kaggle/kaggle.json, that you can generate from "Create new API token"
# on your account page in kaggle.com
rm -rf docs/user/advanced_examples/local_datasets
mkdir docs/user/advanced_examples/local_datasets
cd docs/user/advanced_examples/local_datasets

kaggle competitions download -c titanic
unzip titanic.zip -d titanic
