<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/deployment/deploy_to_docker.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.deploy_to_docker`

Methods to deploy a server using Docker.

It takes as input a folder with:
\- client.zip
\- server.zip
\- processing.json

It builds a Docker image and spawns a Docker container that runs the server.

This module is untested as it would require to first build the release Docker image. FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3347

## **Global Variables**

- **DATE_FORMAT**

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_docker.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `delete_image`

```python
delete_image(image_name: str)
```

Delete a Docker image.

**Arguments:**

- <b>`image_name`</b> (str):  to name of the image to delete.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_docker.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stop_container`

```python
stop_container(image_name: str)
```

Kill all containers that use a given image.

**Arguments:**

- <b>`image_name`</b> (str):  name of Docker image for which to stop Docker containers.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_docker.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_docker_image`

```python
build_docker_image(path_to_model: Path, image_name: str)
```

Build server Docker image.

**Arguments:**

- <b>`path_to_model`</b> (Path):  path to serialized model to serve.
- <b>`image_name`</b> (str):  name to give to the image.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_docker.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `main`

```python
main(path_to_model: Path, image_name: str)
```

Deploy function.

- Builds Docker image.
- Runs Docker server.
- Stop container and delete image.

**Arguments:**

- <b>`path_to_model`</b> (Path):  path to model to server
- <b>`image_name`</b> (str):  name of the Docker image
