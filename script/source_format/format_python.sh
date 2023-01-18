#!/usr/bin/env bash

function usage() {
    echo "$0: install system and data, to support compiler"
    echo
    echo "--help                    Print this message"
    echo "--check                   Do not apply format"
    echo "--dir                     Specify a source directory"
    echo
}

CHECK=

while [ -n "$1" ]
do
   case $1 in
        "--help" | "-h" )
            usage
            exit 0
            ;;

        "--check" )
            CHECK="$1"
            ;;

        "--dir" )
            shift
            DIRS+=("$1")
            ;;

        "--file" )
            shift
            FILES+=("$1")
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

for SRC_DIR in "${DIRS[@]}"; do
    isort -l 100 --profile black ${CHECK:+"$CHECK"} "${SRC_DIR}"
    ((FAILURES+=$?))
    black -l 100 ${CHECK:+"$CHECK"} "${SRC_DIR}"
    ((FAILURES+=$?))
done

for SRC in "${FILES[@]}"; do
    isort -l 100 --profile black ${CHECK:+"$CHECK"} "${SRC}"
    ((FAILURES+=$?))
    black -l 100 ${CHECK:+"$CHECK"} "${SRC}"
    ((FAILURES+=$?))
done

if [[ "$FAILURES" != "0" ]]; then
    exit 1
fi
