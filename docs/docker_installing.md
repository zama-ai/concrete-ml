# Using Docker Image

You can also get the **concrete-ml** Docker image by either pulling the latest Docker image or a specific version:

```shell
docker pull zamafhe/concrete-ml:latest
# or
docker pull zamafhe/concrete-ml:v0.3.0
```

Of course, in commandlines depicted in this page, you will replace `v0.3.0` by the proper version.

The image can be used with Docker volumes, [see the Docker documentation here](https://docs.docker.com/storage/volumes/).

You can then use this image with the following command:

```shell
# Without local volume:
docker run --rm -it -p 8888:8888 zamafhe/concrete-ml

# With local volume to save notebooks on host:
docker run --rm -it -p 8888:8888 -v /host/path:/data zamafhe/concrete-ml
```

This will launch a **Concrete-ML** enabled Jupyter server in Docker, that you can access from your browser.

Alternatively, you can just open a shell in Docker with or without volumes:

```shell
docker run --rm -it zamafhe/concrete-ml /bin/bash
```
