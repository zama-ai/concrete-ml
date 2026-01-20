# Private Authentication in FHE

This folder contains an example that shows how to combine Concrete ML and TFHE-rs workflows through a privacy-preserving, server-side authentication scenario. In this example, access to a remote server is granted via a token, only if the client's encrypted information meets specific criteria. The returned token thus serves as proof of successful authentication — all without ever exposing any sensitive data.

## Overview

To determine whether the client meets the specified requirements, the problem is treated as a classification task. In this example, a decision tree model is used for that purpose and Concrete ML handles the encrypted inference part, while training is performed on clear data.

The process involves the following steps:

**On the developer side**:

1. Train the decision tree model on  clear data
1. Compile the model to an FHE circuit using the TFHE-rs ciphertext format
1. Deploy the model using Concrete ML APIs. To know more about it, please refer to the  [Client-Server Guide](../../docs/guides/client_server.md) and the [Client-Server Notebook](../../docs/advanced_examples/ClientServer.ipynb)

**On the client Side**:

- Encrypt the client's information using a private key and send it to the server.

**On the server side**:

1. Use Concrete ML to predict whether the client's information is valid. If the prediction is positive, the user is authenticated; otherwise, authentication is denied.
1. Use TFHE-rs for the post-processing part to generate a random token (known only to the server) and multiply it by the output of the decision.
1. Return the result, the encrypted token if the user is authenticated; otherwise, an encrypted zero vector.

**On the client side**:

- Decrypt the server's response and send it back to the server to finalize the authentication process.

## Installation

- First, create a virtual env and activate it:

<!--pytest-codeblocks:skip-->

```bash
python -m venv .venv
source .venv/bin/activate
```

- Then, install required packages:

<!--pytest-codeblocks:skip-->

```bash
pip install -r requirements.txt --ignore-installed
```
