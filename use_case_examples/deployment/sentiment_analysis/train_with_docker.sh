#! /bin/env bash

bash download_data.sh && \
  docker build --tag train_sentiment_analysis --file Dockerfile.train . && \
  docker run --name train_sentiment_analysis_container train_sentiment_analysis && \
  docker cp train_sentiment_analysis_container:/project/dev . &&
  docker cp train_sentiment_analysis_container:/project/hf_cache .

docker rm "$(docker ps -a --filter name=train_sentiment_analysis_container -q)"
