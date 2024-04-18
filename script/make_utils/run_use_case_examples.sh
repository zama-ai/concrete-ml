#!/usr/bin/env bash
set -e

CURRENT_DIR=$(dirname "$(realpath "$0")")/../../
USE_CASE_DIR_NAME="use_case_examples"
USE_CASE_DIR="${CURRENT_DIR}/${USE_CASE_DIR_NAME}"

export USE_CASE_DIR  # Required for the Makefile of the use case examples

# Check if a directory exists
check_directory_exists() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory '${1}' not found."
        exit 1
    fi
}

# Check if the git status is clean
check_clean_git_status() {
    if git ls-files --others --exclude-standard | grep -q "$1"; then
        echo "Error: The repository is not clean. Untracked files found in $1."
        echo "List untracked files with: git ls-files --others --exclude-standard | grep $1"
        echo "Remove untracked files with: git clean -fdx $1"
        exit 1
    fi
}

# Setup a virtual environment for a specific use case example
setup_virtualenv() {
    local venv_path="/tmp/virtualenv_$1"
    echo "Setting up virtual environment in $venv_path..."
    python3 -m venv "$venv_path"
    # shellcheck disable=SC1091,SC1090
    source "${venv_path}/bin/activate"
    echo "Virtual environment activated."
}

# Install requirements for a specific use case example
install_requirements() {
    pip install -U pip setuptools wheel
    local example_dir=$1
    if [ -f "${example_dir}/requirements.txt" ]; then
        pushd "$example_dir"
        if pip install -r requirements.txt; then
            echo "Requirements installed successfully."
        else
            echo "Failed to install requirements."
            popd
            return 1
        fi
        popd
    fi
}

# Run a specific use case example
run_example() {
    local example_dir=$1
    local example_name
    example_name=$(basename "$example_dir")

    if [ ! -f "${example_dir}/Makefile" ]; then
        echo "No Makefile found in $example_dir, skipping..."
        return
    fi

    echo "*** Running example: $example_name ***"
    setup_virtualenv "$example_name"
    install_requirements "$example_dir" || return
    set +e
    echo "Running use case example using Makefile..."
    make -C "$example_dir" run_example
    local result=$?
    set -e

    if [ "$result" -ne 0 ]; then
        echo "Failure in example $example_name."
        failed_examples+=("$example_name")
    else
        echo "Successfully completed example $example_name."
        success_examples+=("$example_name")
    fi

    deactivate
    rm -rf "/tmp/virtualenv_$example_name"
}

# Print the summary of execution results
print_summary() {
    echo "Summary of execution results:"
    echo "Successful examples: ${#success_examples[@]}"
    for example in "${success_examples[@]}"; do
        echo "  - $example"
    done
    echo "Failed examples: ${#failed_examples[@]}"
    for example in "${failed_examples[@]}"; do
        echo "  - $example"
    done
}

# Main function to run use case examples
main() {
    check_directory_exists "$USE_CASE_DIR"
    check_clean_git_status "$USE_CASE_DIR_NAME"

    declare -a success_examples
    declare -a failed_examples

    local LIST_OF_USE_CASES=()
    # shellcheck disable=SC2153
    if [[ -z "${USE_CASE}" ]]; then
        mapfile -t LIST_OF_USE_CASES < <(find "$USE_CASE_DIR/" -mindepth 1 -maxdepth 2 -type d | grep -v checkpoints | sort)
    else
        LIST_OF_USE_CASES=("${USE_CASE_DIR}/${USE_CASE}")
    fi

    for use_case in "${LIST_OF_USE_CASES[@]}"; do
        run_example "$use_case"
    done

    print_summary

    if [ ${#failed_examples[@]} -ne 0 ]; then
        exit 1
    fi
}

main "$@"
