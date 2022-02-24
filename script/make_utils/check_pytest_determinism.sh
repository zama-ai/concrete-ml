#!/bin/bash

set -e

RANDOMLY_SEED=$RANDOM
echo "Testing determinism with seed $RANDOMLY_SEED"

OUTPUT_DIRECTORY=$(mktemp -d)

set +e

RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/one.txt"

# This would not be readable
# SC2181: Check exit code directly with e.g. 'if mycmd;', not indirectly with $?.
# shellcheck disable=SC2181
if [ $? -ne 0 ]
then
    echo "The commandline failed with:"
    cat  "${OUTPUT_DIRECTORY}/one.txt"
    exit 255
fi

RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/two.txt"

# This would not be readable
# SC2181: Check exit code directly with e.g. 'if mycmd;', not indirectly with $?.
# shellcheck disable=SC2181
if [ $? -ne 0 ]
then
    echo "The commandline failed with:"
    cat  "${OUTPUT_DIRECTORY}/two.txt"
    exit 255
fi

set -e

# Exceptions:
#   passed in: since it is related to timings
diff "${OUTPUT_DIRECTORY}/one.txt" "${OUTPUT_DIRECTORY}/two.txt" -I "passed in"
echo "Successful determinism check"
