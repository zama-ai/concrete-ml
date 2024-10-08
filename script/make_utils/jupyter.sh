#!/usr/bin/env bash
set -e

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

print_time_execution() {
    printf "Notebook %s executed in %02dh:%02dm:%02ds\n" "$1" $(($2/3600)) $(($2%3600/60)) $(($2%60))
}


WHAT_TO_DO="open"

# Create a list of notebooks to skip, usually because of their long execution time.
# Deployment notebook is currently failing
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4064
NOTEBOOKS_TO_SKIP=("docs/advanced_examples/Deployment.ipynb")

while [ -n "$1" ]
do
   case "$1" in
        "--run_all_notebooks" )
            WHAT_TO_DO="run_all_notebooks"
            ;;

        "--run_all_notebooks_gpu" )
            export CML_USE_GPU=1
            WHAT_TO_DO="run_all_notebooks_gpu"
            ;;

        "--run_all_notebooks_parallel" )
            WHAT_TO_DO="run_all_notebooks_parallel"
            ;;

        "--run_notebook" )
            WHAT_TO_DO="run_notebook"
            shift
            NOTEBOOK="$1"
            ;;

        "--open" )
            WHAT_TO_DO="open"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

if [ "$WHAT_TO_DO" == "open" ]
then
    if isDockerContainer; then
        poetry run jupyter notebook --allow-root --no-browser --ip=0.0.0.0
    else
        poetry run jupyter notebook --allow-root
    fi
elif [ "$WHAT_TO_DO" == "run_all_notebooks" ] || [ "$WHAT_TO_DO" == "run_all_notebooks_gpu" ]
then
    echo "Refreshing notebooks"

    SUCCESSFUL_NOTEBOOKS="./successful_notebooks.txt"
    FAILED_NOTEBOOKS="./failed_notebooks.txt"
    echo "" > "${SUCCESSFUL_NOTEBOOKS}"
    echo "" > "${FAILED_NOTEBOOKS}"

    # shellcheck disable=SC2207
    if [ "$WHAT_TO_DO" == "run_all_notebooks_gpu" ]
    then
        LIST_OF_NOTEBOOKS=($(grep -Rnwl './docs/advanced_examples/' --include \*.ipynb -e 'device =' | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_checkpoints"))
    else
        LIST_OF_NOTEBOOKS=($(find ./docs -type f -name "*.ipynb" | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_checkpoints"))
    fi

    # Remove notebooks with long execution times
    for NOTEBOOK_TO_REMOVE in "${NOTEBOOKS_TO_SKIP[@]}"
    do
        echo "Skipping: ${NOTEBOOK_TO_REMOVE}"

        # shellcheck disable=SC2206
        LIST_OF_NOTEBOOKS=(${LIST_OF_NOTEBOOKS[@]/*${NOTEBOOK_TO_REMOVE}*/})
    done

    echo ""

    # shellcheck disable=SC2068
    for NOTEBOOK in ${LIST_OF_NOTEBOOKS[@]}
    do
        echo "Refreshing ${NOTEBOOK}"

        START=$(date +%s)
        if jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}"; then
            echo "${NOTEBOOK}" >> "${SUCCESSFUL_NOTEBOOKS}"
        else
            echo "${NOTEBOOK}" >> "${FAILED_NOTEBOOKS}"
        fi
        END=$(date +%s)
        TIME_EXEC=$((END-START))
        print_time_execution "${NOTEBOOK}" ${TIME_EXEC}
    done

    # Then, one needs to sanitize the notebooks
elif [ "$WHAT_TO_DO" == "run_all_notebooks_parallel" ]
then
    echo "Refreshing notebooks in parallel"

    # shellcheck disable=SC2207
    LIST_OF_NOTEBOOKS=($(find ./docs/ -type f -name "*.ipynb" | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_checkpoints"))

    # Remove notebooks with long execution times
    for NOTEBOOK_TO_REMOVE in "${NOTEBOOKS_TO_SKIP[@]}"
    do
        echo "${NOTEBOOK_TO_REMOVE} is skipped as its execution time is too long"

        # shellcheck disable=SC2206
        LIST_OF_NOTEBOOKS=(${LIST_OF_NOTEBOOKS[@]/*${NOTEBOOK_TO_REMOVE}*/})
    done

    PIDS_TO_WATCH=""

    # Run notebooks in sub processes on the same machine 
    # shellcheck disable=SC2068
    for NOTEBOOK in ${LIST_OF_NOTEBOOKS[@]}
    do
        ( jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}" ) &
        # store PID of process
        NOTEBOOK_PID="$!"
        PIDS_TO_WATCH+=" ${NOTEBOOK_PID}"
        echo "Notebook ${NOTEBOOK} refreshing with PID: ${NOTEBOOK_PID}"
    done

    STATUS=0
    for p in ${PIDS_TO_WATCH}; do
        if wait "${p}"; then
            echo "Process ${p} success"
        else
            echo "Process ${p} fail"
            STATUS=1
        fi
    done

    exit "${STATUS}"

elif [ "$WHAT_TO_DO" == "run_notebook" ]
then
    START=$(date +%s)
    jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}"
    END=$(date +%s)
    TIME_EXEC=$((END-START))
    print_time_execution "${NOTEBOOK}" ${TIME_EXEC}
    # Then, one needs to sanitize the notebooks
fi


