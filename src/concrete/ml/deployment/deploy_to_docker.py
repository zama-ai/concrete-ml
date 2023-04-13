"""Methods to deploy a server using Docker.

It takes as input a folder with:
    - client.zip
    - server.zip
    - processing.json

It builds a Docker image and spawns a Docker container that runs the server.

This module is untested as it would require to first build the release Docker image.
FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3347
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

DATE_FORMAT: str = "%Y_%m_%d_%H_%M_%S"


def delete_image(image_name: str):
    """Delete a Docker image.

    Arguments:
        image_name (str): to name of the image to delete.
    """
    to_delete = subprocess.check_output(
        f"docker ps -a --filter name={image_name} -q", shell=True
    ).decode("utf-8")
    if to_delete:
        subprocess.check_output(f"docker rmi {to_delete}", shell=True)


def stop_container(image_name: str):
    """Kill all containers that use a given image.

    Arguments:
        image_name (str): name of Docker image for which to stop Docker containers.
    """
    to_delete = subprocess.check_output(
        f"docker ps -q --filter ancestor={image_name}", shell=True
    ).decode("utf-8")
    if to_delete:
        subprocess.check_output(f"docker kill {to_delete}", shell=True)


def build_docker_image(path_to_model: Path, image_name: str):
    """Build server Docker image.

    Arguments:
        path_to_model (Path): path to serialized model to serve.
        image_name (str): name to give to the image.
    """
    delete_image(image_name)

    path_of_script = Path(__file__).parent.resolve()

    cwd = os.getcwd()
    with TemporaryDirectory() as directory:
        temp_dir = Path(directory)

        files = ["server.py", "server_requirements.txt"]
        # Copy files
        for file_name in files:
            source = path_of_script / file_name
            target = temp_dir / file_name
            shutil.copyfile(src=source, dst=target)
        shutil.copytree(path_to_model, temp_dir / "dev")

        # Build image
        os.chdir(temp_dir)
        command = (
            f'docker build --tag {image_name}:latest --file "{path_of_script}/Dockerfile.server" .'
        )
        subprocess.check_output(command, shell=True)
    os.chdir(cwd)


def main(path_to_model: Path, image_name: str):
    """Deploy function.

    - Builds Docker image.
    - Runs Docker server.
    - Stop container and delete image.

    Arguments:
        path_to_model (Path): path to model to server
        image_name (str): name of the Docker image
    """

    build_docker_image(path_to_model, image_name)

    if args.only_build:
        return

    # Run newly created Docker server
    try:
        with open("./url.txt", mode="w", encoding="utf-8") as file:
            file.write("http://localhost:5000")
        subprocess.check_output(f"docker run -p 5000:5000 {image_name}", shell=True)
    except KeyboardInterrupt:
        message = "Terminate container? (y/n) "
        shutdown_instance = input(message).lower()
        while shutdown_instance not in {"no", "n", "yes", "y"}:
            shutdown_instance = input(message).lower()
        if shutdown_instance in {"y", "yes"}:
            stop_container(image_name=image_name)
            delete_image(image_name=image_name)
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-model", dest="path_to_model", type=Path, default=Path("./dev"))
    parser.add_argument("--image-name", dest="image_name", type=Path, default="server")
    parser.add_argument("--only-build", dest="only_build", action="store_true")
    args = parser.parse_args()
    main(path_to_model=args.path_to_model, image_name=args.image_name)
