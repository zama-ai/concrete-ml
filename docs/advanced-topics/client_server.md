# Production Deployment

Concrete-ML provides functionality to deploy FHE machine learning models in a client/server setting. The deployment workflow and model serving follows the following pattern:

## Deployment

The training of the model and its compilation to FHE are performed on a development machine. Three different files are created when saving the model:

- `client.json`; contains the secure cryptographic parameters needed for the client to generate the private and evaluation keys
- `server.json`; contains the compiled model. This file is sufficient to run the model on a server.
- `serialized_processing.json`; contains the metadata about the pre and post processing, such as quantization parameters to quantize the input and dequantize the output.

The compiled model (`server.zip`) is deployed to a server and the cryptographic parameters (`client.zip`) along with the model meta data (`serialized_processing.json`) are shared with the clients.

## Serving

The client obtains the cryptographic parameters (using `client.zip`) and generates a private encryption/decryption key as well as a set of public evaluation keys. The public evaluation keys are then sent to the server, while the secret key remains on the client.

The private data is then encrypted using `serialized_processing.json` by the client and sent to the server. Server-side, the FHE model inference is ran on the encrypted inputs using the public evaluation keys.

The encrypted result is then returned by the server to the client, which decrypts it using its private key. Finally, the client performs any necessary post-processing of the decrypted result using `serialized_processing.json`.

## Example notebook

For a complete example, see [this notebook](https://github.com/zama-ai/concrete-ml/tree/main/docs/advanced_examples/ClientServer.ipynb)
