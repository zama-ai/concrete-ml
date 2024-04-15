# Set up the project

Concrete ML is a `Python` library, so `Python` should be installed to develop Concrete ML. `v3.8` and `v3.9` are the only supported versions. Concrete ML also uses `Poetry` and `Make`.

First of all, you need to `git clone` the project:

<!--
```bash
cd /tmp
rm -rf concrete-ml
```
-->

<!--pytest-codeblocks:cont-->

```bash
git clone https://github.com/zama-ai/concrete-ml
```

In order to be able to run all documentation examples, we recommend to also [install git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) and then pull the necessary files :

<!--pytest-codeblocks:skip-->

```bash
git lfs pull
```

On the contrary, to disable downloading all these files (which represents up to several hundreds of MB) when cloning the repository, simply run :

<!--pytest-codeblocks:skip-->

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/zama-ai/concrete-ml
```

## Automatic installation

A simple way to have everything installed is to use the development Docker (see the [Docker setup](docker_setup.md) guide). On Linux and macOS, you have to run the script in `./script/make_utils/setup_os_deps.sh`. Specify the `--linux-install-python` flag if you want to install python3.8 as well on apt-enabled Linux distributions. The script should install everything you need for Docker and bare OS development (you can first review the content of the file to check what it will do).

{% hint style="danger" %}
For Windows users, the `setup_os_deps.sh` script does not install dependencies because of how many different installation methods there are due to the lack of a single package manager.

The first step is to [install Python](project_setup.md#python) (as some of the dev tools depend on it), then [Poetry](project_setup.md#poetry). In addition to installing Python, you are still going to need the following software available on path on Windows, as some of the basic dev tools depend on them:

- git [https://gitforwindows.org/](https://gitforwindows.org/)
- jq [https://github.com/stedolan/jq/releases](https://github.com/stedolan/jq/releases)
- make [https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make)

Development on Windows only works with the Docker environment. Follow [this link to setup the Docker environment](docker_setup.md).
{% endhint %}

## Manual installation

### Python

To manually install Python, you can follow [this](https://realpython.com/installing-python/) guide (alternatively, you can google `how to install Python 3.8 (or 3.9)`).

### Poetry

`Poetry` is used as the package manager. It drastically simplifies dependency and environment management. You can follow [this](https://python-poetry.org/docs/#installation) official guide to install it.

### make

The dev tools use `make` to launch various commands.

On Linux, you can install `make` from your distribution's preferred package manager.

On macOS, you can install a more recent version of `make` via brew:

```shell
# check for gmake
which gmake

# If you don't have it, it will error out, install gmake
brew install make

# recheck, now you should have gmake
which gmake
```

It is possible to install `gmake` as `make`. Check this [StackOverflow post](https://stackoverflow.com/questions/38901894/how-can-i-install-a-newer-version-of-make-on-mac-os) for more info.

On Windows, check [this GitHub gist](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make).

{% hint style="danger" %}
In the following sections, be sure to use the proper `make` tool for your system: `make`, `gmake`, or other.
{% endhint %}

### Cloning the repository

To get the source code of Concrete ML, clone the code repository using the link for your favorite communication protocol (ssh or https).

## Setting up environment on your host OS

We are going to make use of virtual environments. This helps to keep the project isolated from other `Python` projects in the system. The following commands will create a new virtual environment under the project directory and install dependencies to it.

{% hint style="danger" %}
The following command will not work on Windows if you don't have Poetry >= 1.2.
{% endhint %}

```shell
cd concrete-ml
make setup_env
```

### Activating the environment

Finally, activate the newly created environment using the following command:

#### macOS or Linux

```shell
source .venv/bin/activate
```

#### Windows

```shell
source .venv/Scripts/activate
```

### Setting up environment on Docker

Docker automatically creates and sources a venv in ~/dev_venv/

The venv persists thanks to volumes. It also creates a volume for ~/.cache to speedup later reinstallations. You can check which Docker volumes exist with:

```shell
docker volume ls
```

You can still run all `make` commands inside Docker (to update the venv, for example). Be mindful of the current venv being used (the name in parentheses at the beginning of your command prompt).

```shell
# Here we have dev_venv sourced
(dev_venv) dev_user@8e299b32283c:/src$ make setup_env
```

### Leaving the environment

After your work is done, you can simply run the following command to leave the environment:

```shell
deactivate
```

### Syncing environment with the latest changes

From time to time, new dependencies will be added to the project or the old ones will be removed. The command below will make sure the project has the proper environment, so run it regularly!

```shell
make sync_env
```

### Troubleshooting your environment

#### in your OS

If you are having issues, consider using the dev Docker exclusively (unless you are working on OS-specific bug fixes or features).

Here are the steps you can take on your OS to try and fix issues:

```shell
# Try to install the env normally
make setup_env

# If you are still having issues, sync the environment
make sync_env

# If you are still having issues on your OS, delete the venv:
rm -rf .venv

# And re-run the env setup
make setup_env
```

At this point, you should consider using Docker as nobody will have the exact same setup as you. If, however, you need to develop on your OS directly, you can [ask Zama for help](../README.md#support-channels).

#### in Docker

Here are the steps you can take in your Docker to try and fix issues:

```shell
# Try to install the env normally
make setup_env

# If you are still having issues, sync the environment
make sync_env

# If you are still having issues in Docker, delete the venv:
rm -rf ~/dev_venv/*

# Disconnect from Docker
exit

# And relaunch, the venv will be reinstalled
make docker_start

# If you are still out of luck, force a rebuild which will also delete the volumes
make docker_rebuild

# And start Docker, which will reinstall the venv
make docker_start
```

If the problem persists at this point, you should ask for help. We're here and ready to assist!
