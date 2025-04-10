# Production Deployment

This document explains the deployment workflow and the model serving pattern for deploying Fully Homomorphic Encryption machine learning models in a client/server setting using Concrete ML.

## Deployment

The steps to prepare a model for encrypted inference in a client/server setting is illustrated as follows:

![](../figures/concretemlgraph1.jpg)

### Model training and compilation

The training of the model and its compilation to FHE are performed on a development machine.

Three different files are created when saving the model:

- `client.zip` contains the following files:
  - `client.specs.json` lists the secure cryptographic parameters needed for the client to generate private and evaluation keys.
  - `serialized_processing.json` describes the pre-processing and post-processing required by the machine learning model, such as quantization parameters to quantize the input and de-quantize the output.
- `server.zip` contains the compiled model. This file is sufficient to run the model on a server. The compiled model is machine-architecture specific, for example, a model compiled on x86 cannot run on ARM.

### Model deployment

The compiled model (`server.zip`) is deployed to a server. The cryptographic parameters (`client.zip`) are shared with the clients. In some settings, such as a phone application, the `client.zip` can be directly deployed on the client device and the server does not need to host it.

{% hint style="info" %}
**Important:** In a client-server production using FHE, the server's output format depends on the model type:

- For regressors, the output matches the `predict()` method from scikit-learn, providing direct predictions.
- For classifiers, the output uses the `predict_proba()` method format, offering probability scores for each class, which allows clients to determine class membership by applying a threshold (commonly 0.5).
  {% endhint %}

### Using the API Classes

The `FHEModelDev`, `FHEModelClient`, and `FHEModelServer` classes in the `concrete.ml.deployment` module simplifies the deployment and interaction between the client and server:

- **`FHEModelDev`**:

  - This class handles the serialization of the underlying FHE circuit as well as the crypto-parameters used for generating the keys.
  - Use the `save` method of this class during the development phase to prepare and save the model artifacts (`client.zip` and `server.zip`). With `save` method, you can deploy a trained model or a [training FHE program](../built-in-models/training.md).

- **`FHEModelClient`** is used on the client side for the following actions:

  - Generate and serialize the cryptographic keys.
  - Encrypt the data before sending it to the server.
  - Decrypt the results received from the server.
  - Load quantization parameters and pre/post-processing from `serialized_processing.json`.

- **`FHEModelServer`** is used on the server side for the following actions:

  - Load the FHE circuit from `server.zip` .
  - Execute the model on encrypted data received from the client.

### Example Usage

```python
from concrete.ml.sklearn import DecisionTreeClassifier
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
import numpy as np

# Define the directory for FHE client/server files
fhe_directory = '/tmp/fhe_client_server_files/'

# Initialize the Decision Tree model
model = DecisionTreeClassifier(n_bits=8)

# Generate some random data for training
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, size=100)

# Train and compile the model
model.fit(X, y)
model.compile(X)

# Setup the development environment
dev = FHEModelDev(path_dir=fhe_directory, model=model)
dev.save()

# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

# Client pre-processes new data
X_new = np.random.rand(1, 20)
encrypted_data = client.quantize_encrypt_serialize(X_new)

# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()

# Server processes the encrypted data
encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

# Client decrypts the result
result = client.deserialize_decrypt_dequantize(encrypted_result)
```

#### Data transfer overview:

- **From Client to Server:** `serialized_evaluation_keys` (once), `encrypted_data`.
- **From Server to Client:** `encrypted_result`.

These objects are serialized into bytes to streamline the data transfer between the client and server.

#### Ciphertext formats and keys

Two types of ciphertext formats are [available in Concrete ML](../getting-started/concepts.md#ciphertext-formats) and both are available for deployment. To use the _TFHE-rs radix_ format, pass the `ciphertext_format` option to the compilation call as follows:

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.common.utils import CiphertextFormat
model.compile(X, ciphertext_format=CiphertextFormat.TFHE_RS)

fhe_directory = '/tmp/fhe_client_server_files_tfhers/'

# Setup the development environment
dev = FHEModelDev(path_dir=fhe_directory, model=model)
dev.save()

# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client_tfhers")
serialized_evaluation_keys, tfhers_evaluation_keys = client.get_serialized_evaluation_keys(include_tfhers_key=True)

# Client pre-processes new data
X_new = np.random.rand(1, 20)
encrypted_data = client.quantize_encrypt_serialize(X_new)

# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()

# Server processes the encrypted data
encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

# Client decrypts the result
result = client.deserialize_decrypt_dequantize(encrypted_result[0])
```

In the example above, a second evaluation key is obtained in the `tfhers_evaluation_keys` variable. This key can be loaded by TFHE-rs Rust programs to perform further computation on the model output ciphertexts.

## Serving

The client-side deployment of a secured inference machine learning model is illustrated as follows:

![](../figures/concretemlgraph3.jpg)

The workflow contains the following steps:

1. **Key generation**: The client obtains the cryptographic parameters stored in `client.zip` and generates a private encryption/decryption key as well as a set of public evaluation keys.
1. **Sending public keys**: The public evaluation keys are sent to the server, while the secret key remains on the client.
1. **Data encryption**: The private data is encrypted by the client as described in the `serialized_processing.json` file in `client.zip`.
1. **Data transmission**: The encrypted data is sent to the server.
1. **Encrypted inference**: Server-side, the FHE model inference is run on encrypted inputs using the public evaluation keys.
1. **Data transmission**: The encrypted result is returned by the server to the client.
1. **Data decryption**: The client decrypts it using its private key.
1. **Post-processing**: The client performs any necessary post-processing of the decrypted result as specified in `serialized_processing.json` (part of `client.zip`).

The server-side implementation of a Concrete ML model is illustrated as follows:

![](../figures/concretemlgraph2.jpg)

The workflow contains the following steps:

1. **Storing the public key**: The public evaluation keys sent by clients are stored.
1. **Model evaluation**: The public evaluation keys are retrieved for the client that is querying the service and used to evaluate the machine learning model stored in `server.zip`.
1. **Sending back the result**: The server sends the encrypted result of the computation back to the client.

## Example notebook

For a complete example, see [the client-server notebook](../advanced_examples/ClientServer.ipynb) or [the use-case examples](../../use_case_examples/deployment/).
