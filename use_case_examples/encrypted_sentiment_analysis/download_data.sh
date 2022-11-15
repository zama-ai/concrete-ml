#!/usr/bin/env bash

set -e

# You need to have a valid ~/.kaggle/kaggle.json, that you can generate from "Create new API token"
# on your account page in kaggle.com
rm -rf local_datasets
mkdir local_datasets
cd local_datasets

kaggle datasets download -d crowdflower/twitter-airline-sentiment

unzip twitter-airline-sentiment.zip -d twitter-airline-sentiment
