#!/usr/bin/env bash
# on m1 mac there is a known issue with grpc install

# https://stackoverflow.com/questions/72620996/apple-m1-symbol-not-found-cfrelease-while-running-python-app/74306400#74306400
# if using conda a conda install seem to work too according to the stackoverflow issue

pip uninstall grpcio
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install grpcio --no-binary :all:
