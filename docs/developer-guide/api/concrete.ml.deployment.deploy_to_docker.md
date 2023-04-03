<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/deployment/deploy_to_docker.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.deploy_to_docker`

Methods to deploy a server using docker.

It takes as input a folder with:
\- client.zip
\- server.zip
\- processing.json

It builds a docker image and spawns a docker container that runs the server.

This module is untested as it would require to first build the release docker image. FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3347

## **Global Variables**

- **DATE_FORMAT**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/deployment/deploy_to_docker.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `delete_image`

```python
delete_image(image_name: str)
```

Delete a docker image.

**Arguments:**

- <b>`image_name`</b> (str):  to name of the image to delete.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/deployment/deploy_to_docker.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stop_container`

```python
stop_container(image_name: str)
```

Kill all containers that use a given image.

**Arguments:**

- <b>`image_name`</b> (str):  name of docker image for which to stop docker containers.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/deployment/deploy_to_docker.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_docker_image`

```python
build_docker_image(path_to_model: Path, image_name: str)
```

Build server docker image.

**Arguments:**

- <b>`path_to_model`</b> (Path):  path to serialized model to serve.
- <b>`image_name`</b> (str):  name to give to the image.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/deployment/deploy_to_docker.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `main`

```python
main(path_to_model: Path, image_name: str)
```

Deploy function.

- Builds docker image.
- Runs docker server.
- Stop container and delete image.

**Arguments:**

- <b>`path_to_model`</b> (Path):  path to model to server
- <b>`image_name`</b> (str):  name of the docker image
