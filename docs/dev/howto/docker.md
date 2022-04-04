# Set Up Docker

## Setting up Docker and X forwarding

Before you start this section, go ahead and install Docker. You can follow [this](https://docs.docker.com/engine/install/) official guide if you require assistance.

X forwarding means redirecting the display to your host machine screen so that the Docker container can display things on your screen (otherwise you would only get CLI/terminal interface to your container).

### Linux

```shell
xhost +localhost
```

### MacOS.

To be able to use X forwarding on macOS:

- Install XQuartz
- Open XQuartz.app and make sure that `authorize network connections` is set in the application parameters (currently in the Security settings)
- Open a new terminal within XQuartz.app and type:

```shell
xhost +127.0.0.1
```

Now, the X server should be all set in Docker (in the regular terminal).

### Windows

Install [Xming](https://sourceforge.net/projects/xming/) and use Xlaunch:

- Multiple Windows, Display number: 0
- `Start no client`
- **IMPORTANT**: Check `No Access Control`
- You can save this configuration to relaunch easily, then click finish.

## Building the image

Once you have access to this repository and the dev environment is installed on your host OS (via `make setup_env` once [you followed the steps here](../../dev/howto/project_setup.md)), you should be able to launch the commands to build the dev Docker image with `make docker_build`.

Once you do that, you can get inside the Docker environment using the following command:

```shell
make docker_start

# or build and start at the same time
make docker_build_and_start
# or equivalently but shorter
make docker_bas
```

After you finish your work, you can leave Docker by using the `exit` command or by pressing `CTRL + D`.
