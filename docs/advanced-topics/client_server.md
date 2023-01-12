# Production Deployment

Concrete-ML provides functionality to deploy FHE machine learning models in a client/server setting. The deployment workflow and model serving pattern is as follows:

## Deployment

![](../figures/concretemlgraph1.jpg)

The diagram above shows the steps that a developer goes through to prepare a model for encrypted inference
in a client-server setting.  The training of the model and its compilation to FHE are performed on a development machine. Three different files are created when saving the model:

- `client.zip` contains `client.specs.json` which lists the secure cryptographic parameters needed for the client to generate private and evaluation keys.
- `serialized_processing.json` describes the pre-processing and post-processing needed by the machine learning model, such as quantization parameters to quantize the input and de-quantize the output. It should be deployed in the same way as `client.zip`
- `server.zip` contains the compiled model. This file is sufficient to run the model on a server. The compiled model is machine architecture specific (i.e. a model compiled on x86 can not run on ARM).

The compiled model (`server.zip`) is deployed to a server and the cryptographic parameters (`client.zip`) along with the model meta data (`serialized_processing.json`) are shared with the clients. In some settings, such as a phone app,
the `client.zip` can directly be deployed on the client device and the server does not need to host it.

## Serving

![](../figures/concretemlgraph3.jpg)

The client-side deployment of an secured inference machine learning model follows the schema above. First, the client obtains the cryptographic parameters (stored in `client.zip`) and generates a private encryption/decryption key as well as a set of public evaluation keys. The public evaluation keys are then sent to the server, while the secret key remains on the client.

The private data is then encrypted by the client as described in `serialized_processing.json`, and it is then sent to the server. Server-side, the FHE model inference is run on the encrypted inputs using the public evaluation keys.

The encrypted result is then returned by the server to the client, which decrypts it using its private key. Finally, the client performs any necessary post-processing of the decrypted result as specified in `serialized_processing.json`.

![](../figures/concretemlgraph2.jpg)

The server-side implementation of a Concrete-ML model follows the diagram above. The public evaluation keys sent by clients are stored. They are retrieved for the client that is querying the service and used to evaluate the machine learning model stored in `server.zip`. Finally, the server sends the encrypted result of the computation back to the client.

## Example notebook

For a complete example, see [the client-server notebook](https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/docs/advanced_examples/ClientServer.ipynb).
