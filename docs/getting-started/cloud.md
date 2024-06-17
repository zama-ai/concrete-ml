# Working in the cloud

This document illustrate how Concrete ML model and DataFrames are deployed in client/server setting when creating privacy-preserving services in the cloud.

Once compiled to FHE, a Concrete ML model or DataFrame generates machine code that execute prediction, training or pre-processing on encrypted data. During this process, Concrete ML generates [the private encryption keys](concepts.md#cryptography-concepts) and [the pubic evaluation keys](concepts.md#cryptography-concepts).

## Communication protocols

The overall communications protocol to enable cloud deployment of machine learning services can be summarized in the following diagram:

![](../figures/ClientServerDiag.png)

The steps detailed above are:

1. **Model Deployment**: The model developer deploys the compiled machine learning model to the server. This model includes the cryptographic parameters. The server is now ready to provide private inference. Cryptographic parameters and compiled programs for DataFrames are included directly in Concrete ML.

1. **Client request**: The client requests the cryptographic parameters (client specs). Once the client receives them from the server, the _secret_ and _evaluation_ keys are generated.

1. **Key exchanges**: The client sends the _evaluation_ key to the server. The server is now ready to accept requests from this client. The client sends their encrypted data. Serialized DataFrames include client evaluation keys.

1. **Private inference**: The server uses the _evaluation_ key to securely run prediction, training and pre-processing on the user's data and sends back the encrypted result.

1. **Decryption**: The client now decrypts the result and can send back new requests.

For more information on how to implement this basic secure inference protocol, refer to the [Production Deployment section](../guides/client_server.md) and to the [client/server example](../advanced_examples/ClientServer.ipynb). For information on training on encrypted data, see [the corresponding section](../built-in-models/training.md).
