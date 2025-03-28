#!/usr/bin/env bash

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

LINUX_INSTALL_PYTHON=0
ONLY_LINUX_ACTIONLINT=0

while [ -n "$1" ]
do
   case "$1" in
        "--linux-install-python" )
            LINUX_INSTALL_PYTHON=1
            ;;

        "--only-linux-actionlint" )
            ONLY_LINUX_ACTIONLINT=1
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done


linux_install_gitleaks () {
    GITLEAKS_VERSION=8.17.0
    GITLEAKS_LINUX_X64_SHA256=e0e1d641cc55bcf3c0ecc1abcfc6b432e86611a53121d87ce40eacd9467f98c3

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
        ${SUDO_BIN:+$SUDO_BIN} cp "${GITLEAKS_BIN}" /usr/local/bin/
    else
        echo "Hash mismatch"
        echo "Got sha256:           ${SHA256_DOWNLOADED_FILE}"
        echo "Expected sha256:      ${GITLEAKS_LINUX_X64_SHA256}"
        STATUS=1
    fi
    rm -rf "${TMP_WORKDIR}"
    return "${STATUS}"
}

linux_install_actionlint () {
    ACTIONLINT_VERSION=1.6.12
    ACTIONLINT_LINUX_X64_SHA256=9a7ea97e07a2c058756609274126e78b60ed15c1ed481ccb9da94b74c24d5f3f

    ACTIONLINT_ARCHIVE_LINK="https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz"

    TMP_WORKDIR="$(mktemp -d)"
    DOWNLOADED_FILE="${TMP_WORKDIR}/actionlint.tar.gz"
    wget --https-only --output-document="${DOWNLOADED_FILE}" "${ACTIONLINT_ARCHIVE_LINK}"
    SHA256_DOWNLOADED_FILE="$(sha256sum "${DOWNLOADED_FILE}" | cut -d ' ' -f 1)"
    STATUS=0
    if [[ "${SHA256_DOWNLOADED_FILE}" == "${ACTIONLINT_LINUX_X64_SHA256}" ]]; then
        tar -xvf "${DOWNLOADED_FILE}" -C "${TMP_WORKDIR}"
        ACTIONLINT_BIN="${TMP_WORKDIR}/actionlint"
        chmod +x "${ACTIONLINT_BIN}"
        ${SUDO_BIN:+$SUDO_BIN} cp "${ACTIONLINT_BIN}" /usr/local/bin/
    else
        echo "Hash mismatch"
        echo "Got sha256:           ${SHA256_DOWNLOADED_FILE}"
        echo "Expected sha256:      ${ACTIONLINT_LINUX_X64_SHA256}"
        STATUS=1
    fi
    rm -rf "${TMP_WORKDIR}"
    return "${STATUS}"
}

linux_install_github_cli () {
    # Installs github cli
    # https://github.com/cli/cli/blob/trunk/docs/install_linux.md#debian-ubuntu-linux-raspberry-pi-os-apt
    echo "Installing github-CLI"
    export GH_CLI_VERSION="2.51.0"
    wget -P "$HOME" https://github.com/cli/cli/releases/download/v${GH_CLI_VERSION}/gh_${GH_CLI_VERSION}_linux_amd64.deb
    ${SUDO_BIN:+$SUDO_BIN} dpkg -i "$HOME/gh_${GH_CLI_VERSION}_linux_amd64.deb"
}

linux_install_cmake () {
    # Install lsb-release if not present
    if ! command -v lsb_release >/dev/null 2>&1; then
        ${SUDO_BIN:+$SUDO_BIN} apt-get update
        ${SUDO_BIN:+$SUDO_BIN} apt-get -y install lsb-release
    fi

    # Get Ubuntu version codename
    UBUNTU_CODENAME=$(lsb_release -cs)
    
    # For Ubuntu 22.04 (jammy) and newer, use the default package
    if [[ "${UBUNTU_CODENAME}" == "jammy" || "${UBUNTU_CODENAME}" > "jammy" ]]; then
        ${SUDO_BIN:+$SUDO_BIN} apt-get update
        ${SUDO_BIN:+$SUDO_BIN} apt-get -y install cmake
        return
    fi
    
    # For Ubuntu 20.04 (focal) and older, use Kitware's repository
    ${SUDO_BIN:+$SUDO_BIN} apt-get update
    ${SUDO_BIN:+$SUDO_BIN} apt-get -y install ca-certificates gpg wget
    test -f /usr/share/doc/kitware-archive-keyring/copyright ||
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
        gpg --dearmor - | \
        ${SUDO_BIN:+$SUDO_BIN} tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" | \
        ${SUDO_BIN:+$SUDO_BIN} tee /etc/apt/sources.list.d/kitware.list >/dev/null
    ${SUDO_BIN:+$SUDO_BIN} apt-get update
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
        ${SUDO_BIN:+$SUDO_BIN} rm /usr/share/keyrings/kitware-archive-keyring.gpg
    ${SUDO_BIN:+$SUDO_BIN} apt-get install -y kitware-archive-keyring
    ${SUDO_BIN:+$SUDO_BIN} apt-get update
    ${SUDO_BIN:+$SUDO_BIN} apt-get -y install cmake=3.28.3-0kitware1ubuntu20.04.1 cmake-data=3.28.3-0kitware1ubuntu20.04.1

    cmake --version
}

OS_NAME=$(uname)

if [[ "${OS_NAME}" == "Linux" ]]; then
    # Docker build
    CLEAR_APT_LISTS=""
    SUDO_BIN="$(command -v sudo)"
    if [[ "${SUDO_BIN}" != "" ]]; then
        SUDO_BIN="${SUDO_BIN} "
    fi
   
    PYTHON_PACKAGES=
    if [[ "${LINUX_INSTALL_PYTHON}" == "1" ]]; then
        PYTHON_PACKAGES="python3-pip \
        python3 \
        python3-dev \
        python3-tk \
        python3-venv \
        python-is-python3 \
        "
    fi

    if [[ "${ONLY_LINUX_ACTIONLINT}" == "1" ]]
    then
        SETUP_CMD="linux_install_actionlint"
    else
        SETUP_CMD="${SUDO_BIN:+$SUDO_BIN}apt-get update && \
        ${SUDO_BIN:+$SUDO_BIN}apt-get upgrade --no-install-recommends -y && \
        ${SUDO_BIN:+$SUDO_BIN}apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ${PYTHON_PACKAGES:+$PYTHON_PACKAGES} \
        git \
        git-lfs \
        graphviz* \
        jq \
        make \
        rsync \
        unzip \
        pandoc \
        openssl \
        shellcheck \
        libssl-dev \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-xetex lmodern \
        wget pipx &&
        ${CLEAR_APT_LISTS:+$CLEAR_APT_LISTS} \
        linux_install_gitleaks && \
        linux_install_actionlint && \
        linux_install_github_cli && \
        linux_install_cmake"
    fi
    eval "${SETUP_CMD}"


    # Install poetry, either with pipx for ubuntu >= 23
    # or through regular pip for older ubuntu
    (\
        python3 -m pip install --no-cache-dir --upgrade pip && \
        python3 -m pip install --no-cache-dir --ignore-installed poetry==1.8.4 \
    ) || (pipx install poetry==1.8.4 && pipx ensurepath);

    echo "PATH=$PATH:$HOME/.local/bin/" >> "$HOME/.bashrc"
elif [[ "${OS_NAME}" == "Darwin" ]]; then

    # Some problems with the git which is preinstalled on AWS virtual machines. Let's unlink it
    # but not fail if it is not there, so use 'cat' as a hack to be sure that, even if set -x is
    # activated later in this script, the status is still 0 == success
    brew unlink git@2.35.1 | cat

    brew install curl git git-lfs gitleaks graphviz jq make pandoc shellcheck openssl libomp actionlint unzip gh rsync
    python3 -m pip install -U pip
    python3 -m pip install poetry==1.8.4

    echo "Make is currently installed as gmake"
    echo 'If you need to use it as "make", you can add a "gnubin" directory to your PATH from your bashrc like:'
    # shellcheck disable=SC2016
    echo 'PATH="/usr/local/opt/make/libexec/gnubin:$PATH"'
else
    echo "Unknown OS"
    exit 1
fi
