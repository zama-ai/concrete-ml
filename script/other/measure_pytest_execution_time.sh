#!/usr/bin/env bash

PYTEST_FILES=$(find tests -name "*.py")

OUTPUT_FILE="execution_time_of_individual_pytest_files.txt"

rm -f ${OUTPUT_FILE}

echo "Execution time for:" >> ${OUTPUT_FILE}

for PYTEST_FILE in $PYTEST_FILES
do
    start=$(date +%s)

    poetry run pytest \
    --randomly-dont-reorganize \
    --randomly-dont-reset-seed \
    "${PYTEST_FILE}"

    end=$(date +%s)
    echo "    ${PYTEST_FILE}: $((end - start)) seconds" >> ${OUTPUT_FILE}
done
