#! /bin/env bash
docker build --tag client --file Dockerfile.client . && \
  docker build --tag server --file Dockerfile.server .
