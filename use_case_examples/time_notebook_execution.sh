#!/bin/bash

print_time_execution() {
    printf " - Notebook %s executed in %02dh:%02dm:%02ds\n" "$1" $(($2/3600)) $(($2%3600/60)) $(($2%60))
}


START=$(date +%s)

set +e
jupyter nbconvert --to notebook --inplace --execute "$1" 1>&2
result=$?
set -e

END=$(date +%s)
TIME_EXEC=$((END-START))

# Check the result of nbconvert execution
if [ $result -ne 0 ]; then
    echo "Error: nbconvert failed with status $result"
    exit $result
else
    print_time_execution "$1" ${TIME_EXEC}
fi

