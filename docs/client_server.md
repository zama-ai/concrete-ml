# Deploying FHE models

**Concrete-ML** provides functionality to deploy FHE machine learning models in a client/server setting. The deployment workflow and model serving follows the following pattern:

## Deployment

The training of the model and its compilation to FHE are performed on a development machine. Three different files are created when saving the model:

- client.json; contains the secure cryptographic parameters needed for the client to generate the private and evaluation keys
- server.json; contains the compiled model (This file is enough to run the model on a server)
- serialized_processing.json; contains the meta data about the pre and post processing (essentially quantization parameters to quantize the input and dequantize the output).

The compiled model (`server.zip`) is deployed to a server and the cryptographic parameters(`client.zip`) along with the model meta data (`serialized_processing.json`) are shared with the clients.

## Serving

The client obtains the cryptographic parameters (using `client.zip`) and generates private and evaluation keys. Evaluation keys are sent to the server. Then the private data are encrypted (using `serialized_processing.json`) by the client and sent to the server. The (FHE) model inference is done on the server with the evaluation keys, on those encrypted private data. The encrypted result is returned by the server to the client, which decrypts it using its private key. The client performs any necessary post-processing of the decrypted result (using `serialized_processing.json`).

## Example notebook

We refer the reader to [this notebook](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/ClientServer.ipynb) for a detailed example.
