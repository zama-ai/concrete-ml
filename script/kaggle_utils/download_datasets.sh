#!/usr/bin/env bash

set -e

# You need to have a valid ~/.kaggle/kaggle.json, that you can generate from "Create new API token"
# on your account page in kaggle.com
rm -rf docs/advanced_examples/local_datasets
mkdir docs/advanced_examples/local_datasets
cd docs/advanced_examples/local_datasets

kaggle competitions download -c titanic
kaggle datasets download -d crowdflower/twitter-airline-sentiment

unzip titanic.zip -d titanic
unzip twitter-airline-sentiment.zip -d twitter-airline-sentiment
