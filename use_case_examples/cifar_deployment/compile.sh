#! /bin/env bash

docker build --tag compile_cifar --file Dockerfile.compile . && \
  docker run --name compile_cifar compile_cifar && \
  docker cp compile_cifar:/project/dev . && \
  docker rm "$(docker ps -a --filter name=compile_cifar -q)"
