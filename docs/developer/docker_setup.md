# Set Up Docker

Before you start this section, you must install Docker by following [this](https://docs.docker.com/engine/install/) official guide.

## Building the image

Once you have access to this repository and the dev environment is installed on your host OS (via `make setup_env` once [you followed the steps here](project_setup.md)), you should be able to launch the commands to build the dev Docker image with `make docker_build`.

Once you do that, you can get inside the Docker environment using the following command:

```shell
make docker_start

# or build and start at the same time
make docker_build_and_start

# or equivalently but shorter
make docker_bas
```

After you finish your work, you can leave Docker by using the `exit` command or by pressing `CTRL + D`.
