# Deployment

In this folder we explain how to deploy Concrete ML models.
We show-case how to do this on 3 examples:

- Breast cancer classification using a simple XGBoost model
- Sentiment analysis by running a XGBoost model on top of a Transformer model
- CIFAR-10 classification using a VGG model.

You can run these example locally using Docker.

For all of them the workflow is the same:

1. Optional: Train the model
1. Compile the model to an FHE circuit
1. Deploy to Docker or localhost
1. Run the inference using the client (locally or in Docker)

The script to deploy the model compiled to an FHE circuit is the same for all. The main difference between them is the client. Each use-case needs its own client.
