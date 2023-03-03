# Deployment

In this folder we explain how to deploy CML models.
We show-case how to do this on 3 examples:

- Breast cancer classification using a simple XGBoost model
- Sentiment analysis by running a XGBoost model on top of a Transformer model
- CIFAR-10 classification using a VGG model split in two parts.

You can run these example locally using Docker, or on AWS if you have your credentials set up.

For all of them the workflow is the same:
0\. Optional: Train the model

1. Compile the model to a FHE circuit
1. Deploy to AWS, Docker or localhost
1. Run the inference using the client (locally or in Docker)

The script to deploy the model compiled to a FHE circuit is the same for all. The main difference between them is the client. Each use-case needs its own client.

<!-- 
Needed while 1.x Docker image hasn't been released yet
FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3231
-->

WARNING: Before running these examples you will need to build a Docker image with the current version of the repository.
To do so run the following command from the root of the repository (you will need Poetry that is a development dependency, please refer to the [adequate documentation](../../docs/developer-guide/project_setup.md)):

```
poetry build && mkdir pkg && cp dist/* pkg/ && make release_docker
```
