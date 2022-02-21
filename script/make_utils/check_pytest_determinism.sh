#!/bin/bash

set -e

RANDOMLY_SEED=$RANDOM
echo "Testing determinism with seed $RANDOMLY_SEED"

OUTPUT_DIRECTORY=$(mktemp -d)
RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/one.txt"
RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/two.txt"

# Exceptions:
#   passed in: since it is related to timings
diff "${OUTPUT_DIRECTORY}/one.txt" "${OUTPUT_DIRECTORY}/two.txt" -I "passed in"
echo "Successful determinism check"
