#!/usr/bin/env bash
set -e

current_dir=$(pwd)
use_case_dir_name="use_case_examples"
use_case_dir="${current_dir}/${use_case_dir_name}"

check_directory_exists() {
    if [ ! -d "$1" ]; then
        echo "Error: '$use_case_dir_name' directory must be present in the Concrete ML source root."
        exit 1
    fi
}

check_clean_git_status() {
    if git ls-files --others --exclude-standard | grep -q "$1"; then
        echo "Error: This script must be run in a clean clone of the Concrete ML repo."
        echo "Untracked files detected in $1."
        echo "List untracked files with: git ls-files --others --exclude-standard | grep $1"
        echo "Remove untracked files with: git clean -fdx $1"
        exit 1
    fi
}

setup_virtualenv() {
    local venv_path="/tmp/virtualenv_$1"
    echo "Setting up virtual environment for $1..."
    rm -rf "$venv_path"  # Ensure a clean environment
    python3 -m venv "$venv_path"
    source "${venv_path}/bin/activate"
    echo "Virtual environment created at $venv_path."
}

install_concrete_ml() {
    pip install -U pip setuptools wheel
    pip install -e . || return 1
    echo "Concrete ML installed."
}

install_requirements() {
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt || return 1
        echo "Requirements installed."
    fi
}

run_example() {
    local example_dir=$1
    local example_name=$(basename "$example_dir")

    if [ ! -f "${example_dir}/Makefile" ]; then
        return
    fi

    echo "*** Processing example $example_name ***"
    setup_virtualenv "$example_name"
    cd "$current_dir" || return
    install_concrete_ml || return
    cd "$example_dir" || return
    install_requirements || return

    set +e
    USE_CASE_DIR=$use_case_dir make 3>&2 2>&1 1>&3- | tee /dev/tty | perl -pe 's/\e[^\[\]]*\[.*?[a-zA-Z]|\].*?\a//g'
    local result="${PIPESTATUS[0]}"
    set -e
    if [ "$result" -ne 0 ]; then
        echo "Error while running example $example_name."
        failed_examples+=("$example_name")
    else
        success_examples+=("$example_name")
    fi
    deactivate
    rm -rf "/tmp/virtualenv_$example_name"
}

print_summary() {
    echo
    echo "Summary:"
    echo "Successes: ${#success_examples[@]}"
    for example in "${success_examples[@]}"; do
        echo "  - $example"
    done
    echo "Failures: ${#failed_examples[@]}"
    for example in "${failed_examples[@]}"; do
        echo "  - $example"
    done
}

main() {
    check_directory_exists "$use_case_dir"
    check_clean_git_status "$use_case_dir_name"

    declare -a success_examples
    declare -a failed_examples

    if [[ -z "${USE_CASE}" ]]; then
        LIST_OF_USE_CASES=($(find "$use_case_dir/" -mindepth 1 -maxdepth 2 -type d | grep -v checkpoints | sort))
    else
        LIST_OF_USE_CASES=("${use_case_dir}/${USE_CASE}")
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