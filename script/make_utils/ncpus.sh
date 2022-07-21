#!/usr/bin/env bash

if [[ $(uname) == "Darwin" ]]; then
    sysctl -n hw.logicalcpu
else
    nproc
fi
