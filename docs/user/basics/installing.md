# Installing

## Python package

To install **Concrete-ML** from PyPi, run the following:

```shell
pip install concrete-ml
```

```{note}
Note that **concrete-ml** installs **concrete-numpy** with all extras, including `pygraphviz` to draw graphs.
```

```{WARNING}
`pygraphviz` requires `graphviz` packages being installed on your OS, see <a href="https://pygraphviz.github.io/documentation/stable/install.html">https://pygraphviz.github.io/documentation/stable/install.html</a>
```

```{DANGER}
`graphviz` packages are binary packages that won't automatically be installed by pip.
Do check <a href="https://pygraphviz.github.io/documentation/stable/install.html">https://pygraphviz.github.io/documentation/stable/install.html</a> for instructions on how to install `graphviz` for `pygraphviz`.
```

## Docker image

You can also get the **concrete-ml** Docker image by either pulling the latest Docker image or a specific version:

```shell
docker pull zamafhe/concrete-ml:latest
# or
docker pull zamafhe/concrete-ml:v0.1.0
```

The image can be used with Docker volumes, [see the Docker documentation here](https://docs.docker.com/storage/volumes/).

You can then use this image with the following command:

```shell
# Without local volume:
docker run --rm -it -p 8888:8888 zamafhe/concrete-ml:v0.1.0

# With local volume to save notebooks on host:
docker run --rm -it -p 8888:8888 -v /host/path:/data zamafhe/concrete-ml:v0.1.0
```

This will launch a **Concrete-ML** enabled Jupyter server in Docker, that you can access from your browser.

Alternatively, you can just open a shell in Docker with or without volumes:

```shell
docker run --rm -it zamafhe/concrete-ml:v0.2.0 /bin/bash
```
