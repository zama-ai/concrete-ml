# Set Up the Project

```{note}
You will need to first [install python](#installing-python), this can be done automatically for Linux with the rest of the dependencies running the script indicated below with the `--linux-install-python` flag. If you want to install some of the dependencies manually, we detail the installations of [poetry](#installing-poetry) and [make](#installing-make).

On Linux and macOS you will have to run the script in `./script/make_utils/setup_os_deps.sh`. Specify the `--linux-install-python` flag if you want to install python3.8 as well on apt-enabled Linux distributions. The script should install everything you need for docker and bare OS development (you can first check the content of the file to check what it will do).

It is strongly recommended to use the development docker (see the [docker](../../dev/howto/docker.md) guide). However our helper script should bring all the tools you need to develop directly on Linux and macOS.

For Windows see the Warning admonition below.

The project targets Python 3.8 through 3.9 inclusive.
```

```{warning}
For Windows users the `setup_os_deps.sh` script does not install dependencies, because of how many different installation methods there are/lack of a single package manager.

The first step is to [install python](#installing-python) (as some of our dev tools depend on it), then [poetry](#installing-poetry). In addition to installing python, you are still going to need the following software available on path on Windows, as some of our basic dev tools depend on them:
- git [https://gitforwindows.org/](https://gitforwindows.org/)
- jq [https://github.com/stedolan/jq/releases](https://github.com/stedolan/jq/releases)
- make [https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make)

Development on Windows only works with the docker environment, follow [this link to setup the docker environment](../../dev/howto/docker.md).
```

## Installing Python

**Concrete ML** is a `Python` library, so `Python` should be installed to develop **Concrete ML**. `v3.8` and `v3.9` are the only supported versions.

```{note}
As stated [at the start of this document](#set-up-the-project) you can install python3.8 for Linux automatically if it's available in your distribution's apt repository using the ./script/make_utils/setup_os_deps.sh script.
```

You can follow [this](https://realpython.com/installing-python/) guide to install it (alternatively you can google `how to install python 3.8 (or 3.9)`).

## Installing Poetry

`Poetry` is our package manager. It drastically simplifies dependency and environment management.

```{note}
As stated [at the start of this document](#set-up-the-project) you can install Poetry for macOS and Linux automatically using the ./script/make_utils/setup_os_deps.sh script.
```

You can follow [this](https://python-poetry.org/docs/#installation) official guide to install it.

```{WARNING}
As there is no `concrete-compiler` package for Windows, only the dev dependencies can be installed. This requires poetry >= 1.2.

At the time of writing (March 2022), there is only an alpha version of poetry 1.2 that you can install, use the official installer to install preview versions.
```

## Installing make

The dev tools use `make` to launch the various commands.

```{note}
As stated [at the start of this document](#set-up-the-project) you can install `make` for macOS and Linux automatically if it's available in your distribution's apt repository using the ./script/make_utils/setup_os_deps.sh script.
```

On Linux you can install `make` from your distribution's preferred package manager.

On Mac OS you can install a more recent version of `make` via brew:

```shell
# check for gmake
which gmake
# If you don't have it, it will error out, install gmake
brew install make
# recheck, now you should have gmake
which gmake
```

It is possible to install `gmake` as `make`, check this [StackOverflow post](https://stackoverflow.com/questions/38901894/how-can-i-install-a-newer-version-of-make-on-mac-os) for more info.

On Windows check [this GitHub gist](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make).

```{hint}
In the following sections, be sure to use the proper `make` tool for your system: `make`, `gmake`, or other.
```

## Cloning repository

Now, it's time to get the source code of **Concrete ML**.

Clone the code repository using the link for your favourite communication protocol (ssh or https).

## Setting up environment on your host OS

We are going to make use of virtual environments. This helps to keep the project isolated from other `Python` projects in the system. The following commands will create a new virtual environment under the project directory and install dependencies to it.

```{DANGER}
The following command will not work on Windows if you don't have poetry >= 1.2.
```

```shell
cd concrete-ml
make setup_env
```

## Activating the environment

Finally, all we need to do is to activate the newly created environment using the following command.

### macOS or Linux

```shell
source .venv/bin/activate
```

### Windows

```shell
source .venv/Scripts/activate
```

## Setting up environment on docker

The docker automatically creates and sources a venv in ~/dev_venv/

The venv persists thanks to volumes. We also create a volume for ~/.cache to speed up later reinstallations. You can check which docker volumes exist with:

```shell
docker volume ls
```

You can still run all `make` commands inside the docker (to update the venv, for example). Be mindful of the current venv being used (the name in parentheses at the beginning of your command prompt).

```shell
# Here we have dev_venv sourced
(dev_venv) dev_user@8e299b32283c:/src$ make setup_env
```

## Leaving the environment

After your work is done, you can simply run the following command to leave the environment.

```shell
deactivate
```

## Syncing environment with the latest changes

From time to time, new dependencies will be added to project or the old ones will be removed. The command below will make sure the project has the proper environment. So run it regularly!

```shell
make sync_env
```

## Troubleshooting your environment

### In your OS

If you are having issues, consider using the dev docker exclusively (unless you are working on OS specific bug fixes or features).

Here are the steps you can take on your OS to try and fix issues:

```shell
# Try to install the env normally
make setup_env

# If you are still having issues, sync the environment
make sync_env

# If you are still having issues on your OS delete the venv:
rm -rf .venv

# And re-run the env setup
make setup_env
```

At this point you should consider using docker as nobody will have the exact same setup as you, unless you need to develop on your OS directly, in which case you can ask us for help but may not get a solution right away.

### In docker

Here are the steps you can take in your docker to try and fix issues:

```shell
# Try to install the env normally
make setup_env

# If you are still having issues, sync the environment
make sync_env

# If you are still having issues in docker delete the venv:
rm -rf ~/dev_venv/*

# Disconnect from the docker
exit

# And relaunch, the venv will be reinstalled
make docker_start

# If you are still out of luck, force a rebuild which will also delete the volumes
make docker_rebuild

# And start the docker which will reinstall the venv
make docker_start
```

If the problem persists at this point, you should consider asking for help. We're here and ready to assist!
