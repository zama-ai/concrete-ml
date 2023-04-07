#!/bin/bash

print_time_execution() {
    printf " - Notebook %s executed in %02dh:%02dm:%02ds\n" "$1" $(($2/3600)) $(($2%3600/60)) $(($2%60))
}

START=$(date +%s)
jupyter nbconvert --to notebook --inplace --execute "$1" 1>&2
END=$(date +%s)
TIME_EXEC=$((END-START))
print_time_execution "$1" ${TIME_EXEC}


