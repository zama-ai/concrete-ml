#! /bin/env bash

docker build --tag train --file Dockerfile.train . && \
  docker run --name train_container train && \
  docker cp train_container:/project/dev . &&
  docker cp train_container:/project/hf_cache .

docker rm "$(docker ps -a --filter name=train_container -q)"
