# Installation

{% hint style="danger" %}
Please note that not all hardware/OS combinations are supported. Determine your platform, OS version, and Python version before referencing the table below.
{% endhint %}

Depending on your OS, Concrete-ML may be installed with Docker or with pip:

|               OS / HW                | Available on Docker | Available on pip |
| :----------------------------------: | :-----------------: | :--------------: |
|                Linux                 |         Yes         |       Yes        |
|               Windows                |         Yes         |  Not currently   |
|     Windows Subsystem for Linux      |         Yes         |       Yes        |
|            macOS (Intel)             |         Yes         |       Yes        |
| macOS (Apple Silicon, ie M1, M2 etc) |         Yes         |  Not currently   |

Also, only some versions of `python` are supported: in the current release, these are `3.7` (Linux only), `3.8`, and `3.9`. Moreover, the Concrete-ML Python package requires `glibc >= 2.28`. On Linux, you can check your `glibc` version by running `ldd --version`.

Concrete-ML can be installed on Kaggle ([see question on community for more details](https://community.zama.ai/t/how-do-we-use-concrete-ml-on-kaggle/332)), but not on Google Colab ([see question on community for more details](https://community.zama.ai/t/how-do-i-install-run-concrete-ml-on-google-colab/338)).

Most of these limits are shared with the rest of the Concrete stack (namely Concrete-Numpy and Concrete-Compiler). Support for more platforms will be added in the future.

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

This will automatically install all dependencies, notably Concrete-Numpy.

## Using Docker

Concrete-ML can be installed using Docker by either pulling the latest image or a specific version:

```shell
docker pull zamafhe/concrete-ml:latest
# or
docker pull zamafhe/concrete-ml:v0.4.0
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
