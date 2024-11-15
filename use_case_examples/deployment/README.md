# Deployment Examples

This folder contains examples of how to deploy Concrete ML models using Fully Homomorphic Encryption (FHE). These examples demonstrate the process of training, compiling, and deploying models for various use cases.

## Overview

The deployment process generally follows these steps:

1. Train the model (optional, depending on the use case)
1. Compile the model to an FHE circuit
1. Deploy the model using Docker
1. Run inference using a client (locally or in Docker)

## Available Examples

We provide three different use cases to demonstrate the deployment process:

1. [Breast Cancer Classification](./breast_cancer/README.md)
1. [Sentiment Analysis](./sentiment_analysis/README.md)
1. [CIFAR-10 Image Classification](./cifar/README.md)

## Getting Started

Each example folder contains its own README with specific instructions. However, the general process is similar:

1. Train or compile the model using the provided scripts
1. Deploy the model using `deploy_to_docker.py` from the `server` folder
1. Build the client Docker image
1. Run the client to interact with the deployed model

For detailed instructions, please refer to the README in each example folder.

## Requirements

- Docker
- Python 3.8 or later
- Concrete ML library installed

## Additional Resources

- [Client-Server Guide](../../docs/guides/client_server.md)
- [Server Deployment Scripts](./server/README.md)
