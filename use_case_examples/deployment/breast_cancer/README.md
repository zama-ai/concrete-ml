# Deployment

In this folder we show how to deploy a simple Concrete ML model that does breast cancer classification, either through Docker or Amazon Web Services.

## Get started

To run this example on AWS you will also need to have the AWS CLI properly setup on your system.
To do so please refer to [AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
One can also run this example locally using Docker, or just by running the scripts locally.

#### On the developer machine:

1. To train your model you can

   - use `train_with_docker.sh` to use Docker (recommended way),
   - or, only if you know what you're doing and will manage synchronisation between versions, use `python train.py`

   This will train a model and [serialize the FHE circuit](../../../docs/guides/client_server.md) in a new folder called `./dev`.

#### On the server machine:

1. Copy the './dev' directory from the developer machine.
1. If you need to delete existing Dockers: `docker rm -f $(docker ps -a -q)`
1. Launch the server via:

```
python ../server/deploy_to_docker.py --path-to-model ./dev
```

You will finally see some

> INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)

which means the server is ready to server, on Port 5000.

#### On the client machine:

##### If you go for a Docker part on the client side:

1. Launch the `build_docker_client_image.py` to build a client Docker image.
1. Run the client with `client.sh` script. This will run the container in interactive mode.
1. Then, in this Docker, you can launch the client script to interact with the server:

```
URL="<my_url>" python client.py
```

where `<my_url>` is the content of the `url.txt` file (if you don't set URL, the default is `0.0.0.0`; this defines the IP to use when running server in Docker on localhost).

#### If you go for client side done in Python:

1. Prepare the client side:

```
python3.8 -m venv .venvclient
source .venvclient/bin/activate
pip install -r client_requirements.txt
```

1. Run the client script:

```
URL="http://localhost:8888" python client.py
```

And here it is! Whether you use Docker or Python for the client side, you deployed a Concrete ML model and ran an inference using Fully Homormophic Encryption. In particular, you can see that the FHE predictions are correct.
