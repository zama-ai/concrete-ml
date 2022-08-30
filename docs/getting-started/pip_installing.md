# Installation

## Using PyPi

### Requirements

Installing Concrete-ML using PyPi requires a Linux-based OS or macOS running on an x86 CPU. For Apple Silicon, Docker is the only currently supported option (see [below](pip_installing.md#using-docker)).

Installing on Windows can be done using Docker or WSL. On WSL, Concrete-ML will work as long as the package is not installed in the /mnt/c/ directory, which corresponds to the host OS filesystem.

### Installation

To install Concrete-ML from PyPi, run the following:

```shell
pip install -U pip wheel setuptools
pip install concrete-ml
```

## Using Docker

Concrete-ml can be installed using Docker by either pulling the latest image or a specific version:

```shell
docker pull zamafhe/concrete-ml:latest
# or
docker pull zamafhe/concrete-ml:v0.3.0
```

The image can be used with Docker volumes, [see the Docker documentation here](https://docs.docker.com/storage/volumes/).

The image can then be used via the following command:

```shell
# Without local volume:
docker run --rm -it -p 8888:8888 zamafhe/concrete-ml

# With local volume to save notebooks on host:
docker run --rm -it -p 8888:8888 -v /host/path:/data zamafhe/concrete-ml
```

This will launch a Concrete-ML enabled Jupyter server in Docker that can be accessed directly from a browser.

Alternatively, a shell can be lauched in Docker, with or without volumes:

```shell
docker run --rm -it zamafhe/concrete-ml /bin/bash
```
