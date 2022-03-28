#!/usr/bin/env bash

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

LINUX_INSTALL_PYTHON=0

while [ -n "$1" ]
do
   case "$1" in
        "--linux-install-python" )
            LINUX_INSTALL_PYTHON=1
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done


linux_install_gitleaks () {
    GITLEAKS_VERSION=8.5.2
    GITLEAKS_LINUX_X64_SHA256=d83e4721c58638d5a2128ca70341c87fe78b6275483e7dc769a9ca6fe4d25dfd

    GITLEAKS_ARCHIVE_LINK="https://github.com/zricethezav/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"

    TMP_WORKDIR="$(mktemp -d)"
    DOWNLOADED_FILE="${TMP_WORKDIR}/gitleaks.tar.gz"
    wget --https-only --output-document="${DOWNLOADED_FILE}" "${GITLEAKS_ARCHIVE_LINK}"
    SHA256_DOWNLOADED_FILE="$(sha256sum "${DOWNLOADED_FILE}" | cut -d ' ' -f 1)"
    STATUS=0
    if [[ "${SHA256_DOWNLOADED_FILE}" == "${GITLEAKS_LINUX_X64_SHA256}" ]]; then
        tar -xvf "${DOWNLOADED_FILE}" -C "${TMP_WORKDIR}"
        GITLEAKS_BIN="${TMP_WORKDIR}/gitleaks"
        chmod +x "${GITLEAKS_BIN}"
        cp "${GITLEAKS_BIN}" /usr/local/bin/
    else
        echo "Hash mismatch"
        echo "Got sha256:           ${SHA256_DOWNLOADED_FILE}"
        echo "Expected sha256:      ${GITLEAKS_LINUX_X64_SHA256}"
        STATUS=1
    fi
    rm -rf "${TMP_WORKDIR}"
    return "${STATUS}"
}


OS_NAME=$(uname)

if [[ "${OS_NAME}" == "Linux" ]]; then
    # Docker build
    if isDockerBuildkit || (isDocker && ! isDockerContainer); then
        CLEAR_APT_LISTS="rm -rf /var/lib/apt/lists/* &&"
        SUDO_BIN=""
    else
        CLEAR_APT_LISTS=""
        SUDO_BIN="$(command -v sudo)"
        if [[ "${SUDO_BIN}" != "" ]]; then
            SUDO_BIN="${SUDO_BIN} "
        fi
    fi

    PYTHON_PACKAGES=
    if [[ "${LINUX_INSTALL_PYTHON}" == "1" ]]; then
        PYTHON_PACKAGES="python3-pip \
        python3.8 \
        python3.8-dev \
        python3.8-tk \
        python3.8-venv \
        python-is-python3 \
        "
    fi

    SETUP_CMD="${SUDO_BIN:+$SUDO_BIN}apt-get update && apt-get upgrade --no-install-recommends -y && \
    ${SUDO_BIN:+$SUDO_BIN}apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    ${PYTHON_PACKAGES:+$PYTHON_PACKAGES} \
    git \
    graphviz* \
    jq \
    make \
    pandoc \
    openssl \
    shellcheck \
    wget && \
    ${CLEAR_APT_LISTS:+$CLEAR_APT_LISTS} \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    linux_install_gitleaks"
    eval "${SETUP_CMD}"
elif [[ "${OS_NAME}" == "Darwin" ]]; then
    brew install curl git gitleaks graphviz jq make pandoc shellcheck openssl libomp
    python3 -m pip install -U pip
    python3 -m pip install poetry

    echo "Make is currently installed as gmake"
    echo 'If you need to use it as "make", you can add a "gnubin" directory to your PATH from your bashrc like:'
    # shellcheck disable=SC2016
    echo 'PATH="/usr/local/opt/make/libexec/gnubin:$PATH"'
else
    echo "Unknown OS"
    exit 1
fi
