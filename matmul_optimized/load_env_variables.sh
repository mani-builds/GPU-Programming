#!/usr/bin/env sh

export INCLUDE="${PWD}/src"
export CPATH="${PWD}/src"
export LIBRARY_PATH="${PWD}:${PWD}/src:$LIBRARY_PATH"
export LD_LIBRARY_PATH="${PWD}:$LD_LIBRARY_PATH"
