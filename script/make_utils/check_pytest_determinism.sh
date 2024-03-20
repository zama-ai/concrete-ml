#!/usr/bin/env bash

set -e

RANDOMLY_SEED=$RANDOM
OUTPUT_DIRECTORY=$(mktemp -d)

echo "Testing determinism with seed $RANDOMLY_SEED in folder ${OUTPUT_DIRECTORY}"

set +e

RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/one.txt"

# This would not be readable
# SC2181: Check exit code directly with e.g., 'if mycmd;', not indirectly with $?.
# shellcheck disable=SC2181
if [ $? -ne 0 ]
then
    echo "The commandline failed with:"
    cat  "${OUTPUT_DIRECTORY}/one.txt"
    exit 255
fi

RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/two.txt"

# This would not be readable
# SC2181: Check exit code directly with e.g., 'if mycmd;', not indirectly with $?.
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

# Now, check that one can reproduce conditions of a bug in a single file
# and test without having to relaunch the full pytest

# All lines that start with "tests/seeding/test_seeding.py::" represent a test that passed
LIST_FILES=$(grep "tests/seeding/test_seeding.py::" "${OUTPUT_DIRECTORY}/one.txt")

WHICH=0
echo "" > "${OUTPUT_DIRECTORY}/three.txt"
for x in $LIST_FILES
do
    EXTRA_OPTION=" --randomly-seed=$RANDOMLY_SEED"

    echo "poetry run pytest $x -xsvv $EXTRA_OPTION --randomly-dont-reset-seed"

    # Only take lines after the header, i.e., after line with 'collecting'
    # SC2086 is about double quote to prevent globbing and word splitting, but here, it makes that we have
    # an empty arg in pytest, which is considered as "do pytest for all files"
    # shellcheck disable=SC2086
    poetry run pytest "$x" -xsvv $EXTRA_OPTION --randomly-dont-reset-seed | sed -n -e '/collecting/,$p' | grep -v collecting | grep -v "collected" | grep -v "passed in" | grep -v "PASSED" >> "${OUTPUT_DIRECTORY}/three.txt"

    ((WHICH+=1))
done

# Clean a bit one.txt
sed -n -e '/collecting/,$p' "${OUTPUT_DIRECTORY}/one.txt" | grep -v collecting | grep -v "collected" | grep -v "passed in" | grep -v "PASSED" | grep -v "Leaving directory" > "${OUTPUT_DIRECTORY}/one.modified.txt"

echo ""
echo "diff:"
echo ""
diff -u "${OUTPUT_DIRECTORY}/one.modified.txt" "${OUTPUT_DIRECTORY}/three.txt" --ignore-all-space --ignore-blank-lines --ignore-space-change

# Run an execution and then check that the sub_seeds depends on the file name and the function name and parameters
RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding_system_file_a.py make pytest_one_single_cpu > "${OUTPUT_DIRECTORY}/seeds.txt"

# This would not be readable
# SC2181: Check exit code directly with e.g., 'if mycmd;', not indirectly with $?.
# shellcheck disable=SC2181
if [ $? -ne 0 ]
then
    echo "The commandline failed with:"
    cat  "${OUTPUT_DIRECTORY}/seeds.txt"
    exit 255
fi

RANDOMLY_SEED=$RANDOMLY_SEED TEST=tests/seeding/test_seeding_system_file_b.py make pytest_one_single_cpu >> "${OUTPUT_DIRECTORY}/seeds.txt"

# This would not be readable
# SC2181: Check exit code directly with e.g., 'if mycmd;', not indirectly with $?.
# shellcheck disable=SC2181
if [ $? -ne 0 ]
then
    echo "The commandline failed with:"
    cat  "${OUTPUT_DIRECTORY}/seeds.txt"
    exit 255
fi

# In the following test, we:
#       log all Output's in seeds_outputs.txt, in a file of N lines: then, we'll check that
#           the first Output is different from the 2nd to N-th once
#       log all sub_seed's in seeds_sub_seed.txt, in a file of N lines: then, we'll check that
#           the first sub_seed is different from the 2nd to N-th once
#
# The goal of these test is to check that, really, seeds and outputs are different, ie that the
# filename or the function name or the parameters were really used in the seeding. Ideally, we
# should check that all the outputs and seeds are different from each other, but it would be a long
# test (ie, being in N**2 [or ideally N log(N)] instead of N), so it's not worth it
grep "Output" "${OUTPUT_DIRECTORY}/seeds.txt" > "${OUTPUT_DIRECTORY}/seeds_outputs.txt"
grep "sub_seed" "${OUTPUT_DIRECTORY}/seeds.txt" > "${OUTPUT_DIRECTORY}/seeds_sub_seed.txt"

python script/make_utils/check_first_line_is_different_from_other_lines.py --file "${OUTPUT_DIRECTORY}/seeds_outputs.txt"
python script/make_utils/check_first_line_is_different_from_other_lines.py --file "${OUTPUT_DIRECTORY}/seeds_sub_seed.txt"

echo "Successful final check"
