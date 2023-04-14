#!/usr/bin/env bash

set -e

BASENAME="licenses"
LICENSE_DIRECTORY="deps_licenses"
CHECK=0
DIFF_TOOL="diff --ignore-all-space --ignore-tab-expansion --ignore-space-change --ignore-all-space --ignore-blank-lines --strip-trailing-cr"
TMP_VENV_PATH="/tmp/tmp_venv"
DO_USER_LICENSES=1

OUTPUT_DIRECTORY="${LICENSE_DIRECTORY}"
DO_FORCE_UPDATE=0
CP_VERSION="concrete-python[full]"

while [ -n "$1" ]
do
   case "$1" in
        "--check" )
            CHECK=1
            OUTPUT_DIRECTORY=$(mktemp -d)
            ;;

        "--force_update" )
            DO_FORCE_UPDATE=1
            ;;

        "--cp_version" )
            shift
            CP_VERSION="${1}"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

# Temp file with CP version inside. We can't have a dynamic TMP_FILE_CP_VERSION filename, since it
# changes the md5
TMP_FILE_CP_VERSION="/tmp/cp_version.txt"
rm -f TMP_FILE_CP_VERSION
echo "${CP_VERSION}" > ${TMP_FILE_CP_VERSION}

UNAME=$(uname)
if [ "$UNAME" == "Darwin" ]
then
  if [ "$(uname -m)" == "arm64" ]
  then
    OS=mac_silicon
  else
    OS=mac_intel
  fi
elif [ "$UNAME" == "Linux" ]
then
    OS=linux
else
    echo "Problem with OS"
    exit 255
fi

mkdir -p "${LICENSE_DIRECTORY}"

if [ ${DO_USER_LICENSES} -eq 1 ]
then
    #Licenses for user (install in a temporary venv)
    echo "Doing licenses for user"

    FILENAME="${BASENAME}_${OS}_user.txt"
    LICENSES_FILENAME="${LICENSE_DIRECTORY}/${FILENAME}"
    NEW_LICENSES_FILENAME="${OUTPUT_DIRECTORY}/${FILENAME}"

    # If the dependencies have not changed, don't do anything
    MD5_OLD_DEPENDENCIES=$(cat ${LICENSES_FILENAME}.md5)
    MD5_NEW_DEPENDENCIES=$(openssl md5 poetry.lock ${TMP_FILE_CP_VERSION} | sed -e "s@(stdin)= @@g" | openssl md5 | sed -e "s@MD5@@g" | sed -e "s@(stdin)= @@g")

    echo "MD5 of the poetry.lock for which dependencies have been listed: ${MD5_OLD_DEPENDENCIES}"
    echo "MD5 of the current poetry.lock:                                 ${MD5_NEW_DEPENDENCIES}"

    if [ $CHECK -eq 1 ]
    then
        if [ "${MD5_OLD_DEPENDENCIES}" != "${MD5_NEW_DEPENDENCIES}" ]
        then
            echo "Error, the md5 is not correct"
            exit 255
        fi
    fi

    if [ "${MD5_OLD_DEPENDENCIES}" == "${MD5_NEW_DEPENDENCIES}" ] && [ ${DO_FORCE_UPDATE} -ne 1 ]
    then
        echo "The lock file hasn't changed, early exit (we assume the md5 is correct)"
        exit 0
    fi

    rm -rf $TMP_VENV_PATH/tmp_venv
    python -m venv $TMP_VENV_PATH/tmp_venv

    # Check that we use a recent python (eg, if we were using an old one, it may create a deprecated
    # license file)
    CHECK_VERSION=$(python --version)
    IS_CORRECT_PYTHON=0

    if [[ "${CHECK_VERSION}" == *"3.8"* ]]; then
        IS_CORRECT_PYTHON=1
    elif [[ "${CHECK_VERSION}" == *"3.9"* ]]; then
        IS_CORRECT_PYTHON=1
    elif [[ "${CHECK_VERSION}" == *"3.10"* ]]; then
        IS_CORRECT_PYTHON=1
    fi

    if [ $IS_CORRECT_PYTHON -eq 0 ]
    then
        echo "Error, please don't try to make licenses with an unsupported python version (here: $CHECK_VERSION)"
        exit 255
    else
        echo "Make licenses with python $CHECK_VERSION"
    fi

    # SC1090: Can't follow non-constant source. Use a directive to specify location.
    # shellcheck disable=SC1090,SC1091
    source $TMP_VENV_PATH/tmp_venv/bin/activate

    python -m pip install -U pip wheel

    # Only for linux and docker, reinstall setuptools. On macOS, it creates warnings, see 169
    if [[ "$UNAME" != "Darwin" ]]
    then
        python -m pip install -U --force-reinstall setuptools
    fi

    poetry install --only main
    python -m pip install -U --pre "${CP_VERSION}"
    python -m pip install pip-licenses

    # In --format=csv such that the column length (and so, the diff) do not change with longer
    # names
    pip-licenses --format=csv | tr -d "\"" | grep -v "pkg-resources\|pkg_resources\|concrete-ml," | \
        tee "${NEW_LICENSES_FILENAME}"

    # Remove trailing whitespaces and replace "," by ", "
    if [ "$UNAME" == "Darwin" ]
    then
        sed -i "" 's/[t ]*$//g' "${NEW_LICENSES_FILENAME}"
        sed -i "" 's/,/, /g' "${NEW_LICENSES_FILENAME}"

    else
        sed -i 's/[t ]*$//g' "${NEW_LICENSES_FILENAME}"
        sed -i 's/,/, /g' "${NEW_LICENSES_FILENAME}"
    fi

    # Reject GPL-like stuff
    LICENSES_BLACKLIST="GNU Lesser General Public License v3 (LGPLv3)"
    LICENSES_BLACKLIST="${LICENSES_BLACKLIST};GNU Lesser General Public License v2 (LGPLv2)"
    LICENSES_BLACKLIST="${LICENSES_BLACKLIST};GNU General Public License (GPL)"
    LICENSES_BLACKLIST="${LICENSES_BLACKLIST};GNU General Public License v2 (GPLv2)"
    LICENSES_BLACKLIST="${LICENSES_BLACKLIST};GNU General Public License v3 (GPLv3)"
    LICENSES_BLACKLIST="${LICENSES_BLACKLIST};GNU Library or Lesser General Public License (LGPL)"

    pip-licenses --fail-on="${LICENSES_BLACKLIST}"

    # Check even more GPL things
    IS_THERE_GPL_LICENSES=$(grep GPL "${NEW_LICENSES_FILENAME}" > /dev/null; echo $?)
    if [ "$IS_THERE_GPL_LICENSES" -eq 0 ]
    then
        echo "Problems, there are GPL licences"
        grep GPL "${NEW_LICENSES_FILENAME}"
        exit 255
    fi

    # And check with a white-list
    # Brevitas has an "UNKNOWN" license, but is actually a BSD, so it is ignored in this test
    # pkg-resources reports UNKNOWN due to a Ubuntu bug, but is Apache - ignore
    LICENSES_WHITELIST="new BSD 3-Clause"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};3-Clause BSD License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};new BSD"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};BSD License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};BSD-3"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};BSD 3-Clause"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};BSD-3-Clause"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Apache License v2.0"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Apache 2.0"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Apache Software License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Apache Software License; BSD License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Historical Permission Notice and Disclaimer (HPND)"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};MIT License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};MIT"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};MIT License; Mozilla Public License 2.0 (MPL 2.0)"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Other/Proprietary License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};Python Software Foundation License"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};ISC License (ISCL)"
    LICENSES_WHITELIST="${LICENSES_WHITELIST};The Unlicense (Unlicense)"

    pip-licenses --allow-only="${LICENSES_WHITELIST}" --ignore-packages brevitas pkg-resources pkg_resources concrete-ml-extensions-brevitas

    deactivate

    if [ $CHECK -eq 1 ]
    then
        echo "$DIFF_TOOL $LICENSES_FILENAME ${NEW_LICENSES_FILENAME}"
        $DIFF_TOOL "$LICENSES_FILENAME" "${NEW_LICENSES_FILENAME}"
        echo "Success: no update in $LICENSES_FILENAME"
    else
        # Update the .md5 files
        openssl md5 poetry.lock ${TMP_FILE_CP_VERSION} | sed -e "s@(stdin)= @@g" | openssl md5 | sed -e "s@MD5@@g" | sed -e "s@(stdin)= @@g" > ${LICENSES_FILENAME}.md5
    fi
fi

rm -f ${LICENSE_DIRECTORY}/licenses_*.txt.tmp
rm -rf $TMP_VENV_PATH/tmp_venv

echo "End of license script"
