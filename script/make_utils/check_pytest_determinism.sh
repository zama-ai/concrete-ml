#!/usr/bin/env bash

set -e

RANDOMLY_SEED=$RANDOM
OUTPUT_DIRECTORY=$(mktemp -d)

echo "Testing determinism with seed $RANDOMLY_SEED in folder ${OUTPUT_DIRECTORY}"

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

# Now, check --forcing_random_seed, i.e. check that one can reproduce conditions of a bug in a single file
# and test without having to relaunch the full pytest, by just picking the right --forcing_random_seed

LIST_FILES=$(grep "tests/seeding/test_seeding.py" "${OUTPUT_DIRECTORY}/one.txt")
LIST_SEED=()
while IFS='' read -r line; do LIST_SEED+=("$line"); done < <(grep "forcing_random_seed" "${OUTPUT_DIRECTORY}/one.txt" | sed -e "s@Relaunch the tests with --forcing_random_seed @@" | sed -e "s@ --randomly-dont-reset-seed to reproduce@@")

WHICH=0
echo "" > "${OUTPUT_DIRECTORY}/three.txt"
for x in $LIST_FILES
do
    # For test test_seed_3, we need to add $RANDOMLY_SEED
    EXTRA_OPTION=""
    if [ "$x" == "tests/seeding/test_seeding.py::test_seed_3[random_inputs0]" ]
    then
        EXTRA_OPTION=" --randomly-seed=$RANDOMLY_SEED"
    fi

    echo "poetry run pytest $x -xsvv $EXTRA_OPTION --randomly-dont-reset-seed --forcing_random_seed " "${LIST_SEED[WHICH]}"

    # Only take lines after the header, i.e. after line with 'collecting'
    # SC2086 is about double quote to prevent globbing and word splitting, but here, it makes that we have
    # an empty arg in pytest, which is considered as "do pytest for all files"
    # shellcheck disable=SC2086
    poetry run pytest "$x" -xsvv $EXTRA_OPTION --randomly-dont-reset-seed --forcing_random_seed "${LIST_SEED[WHICH]}" | sed -n -e '/collecting/,$p' | grep -v collecting | grep -v "collected" | grep -v "passed in" | grep -v "PASSED" >> "${OUTPUT_DIRECTORY}/three.txt"

    ((WHICH+=1))
done

# Clean a bit one.txt
sed -n -e '/collecting/,$p' "${OUTPUT_DIRECTORY}/one.txt" | grep -v collecting | grep -v "collected" | grep -v "passed in" | grep -v "PASSED" | grep -v "Leaving directory" > "${OUTPUT_DIRECTORY}/one.modified.txt"

echo ""
echo "diff:"
echo ""
diff -u "${OUTPUT_DIRECTORY}/one.modified.txt" "${OUTPUT_DIRECTORY}/three.txt" --ignore-all-space --ignore-blank-lines --ignore-space-change
echo "Successful --forcing_random_seed check"
