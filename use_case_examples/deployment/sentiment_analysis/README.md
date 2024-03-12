# Deployment

In this folder we show how to deploy a Concrete ML model that does sentiment analysis, either through Docker or Amazon Web Services.
This is based on the sentiment analysis use case example where a XGBoost model is trained on top of a Transformer model.

## Get started

To run this example on AWS you will also need to have the AWS CLI properly setup on your system.
To do so please refer to [AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
One can also run this example locally using Docker, or just by running the scripts locally.

1. To train your model you can use `train.py`, or `train_with_docker.sh` to use Docker (recommended). This operation might take some time.
   This will train a model and [serialize the FHE circuit](../../../docs/guides/client_server.md).
   This will result in a new folder called `./dev`.
1. Once that's done you can use the script provided in Concrete ML in `src/concrete/ml/deployment/`, either use `deploy_to_aws.py` or `deploy_to_docker.py` according to your need.

- `python -m concrete.ml.deployment.deploy_to_docker`
- `python -m concrete.ml.deployment.deploy_to_aws`
  this will create and run a Docker container or an AWS EC2 instance.

3. Once that's done you can launch the `build_docker_client_image.sh` script to build a client Docker image.
1. You can then run the client by using the `client.sh` script. This will run the container in interactive mode.
   To interact with the server you can launch the `client.py` script using `URL="<my_url>" python client.py` where `<my_url>` is the content of the `url.txt` file.

And here it is you deployed a Concrete ML model and ran an inference using Fully Homormophic Encryption.
