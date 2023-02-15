# Deployment

In this folder we show a typical Concrete-ML deployment use-case.
You can start from a trained model, or use `train.py` (or `train.sh` to use docker) to generate the model, and then deploy it to AWS.
To run this example you will need some dependencies like boto3, to install them please run `python -m pip install -r deployment_requirements.txt`.

## Get started

This examples requires you to have the AWS CLI properly setup on your system.
To do so please refer to [AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).

One can also run this example locally using Docker, or just by running the script.

1. First of all you'll need to serialize your model using `FHEModelDev` as explained in [the documentation](../../docs/advanced-topics/client_server.md).
1. Once that's done you can launch the `docker_build_images.sh` script to build both client and server Docker images.
1. Then for AWS deployment run the `deploy_to_aws.py` script. This will create an AWS instance running a FastApi server serving the model. A public IP will be prompted, keep it to use it for the client.
1. To run the client use the `client.sh` script to run the client Docker image. Then launch `URL=http://<public_ip>:5000 python client.py` to launch the inference.
