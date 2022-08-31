# Compilation

Concrete-ML implements machine model inference using Concrete-Numpy as a backend. In order to execute in FHE, a numerical program written in Concrete-Numpy needs to be compiled. This functionality is [described here](https://docs.zama.ai/concrete-numpy/getting-started/compiling_and_executing), and Concrete-ML hides away most of the complexity of this step. The entire compilation process is done by Concrete-Numpy.

From the perspective of the Concrete-ML user, the compilation process performed by Concrete-Numpy can be broken up into 3 steps:

1. numpy program tracing and creation of a Concrete-Numpy op-graph
1. checking that the op-graph is FHE compatible
1. producing machine code for the op-graph. This step automatically determines cryptographic parameters

Additionally, the [client/server API](client_server.md) packages the result of the last step in a way that allows to deploy the encrypted circuit to a server and to perform key generation, encryption and decryption on the client side.

## **Concrete-Numpy** op-graphs and the Virtual Library

The first step in the list above takes a python function implemented using the Concrete-Numpy [supported operation set](https://docs.zama.ai/concrete-numpy/getting-started/numpy_support) and transforms it into an executable operation graph. In this step all the floating point subgraphs in the op-graph are fused and converted to Table Lookup operations.

This enables to:

- execute the op-graph, which includes TLUs, on clear non-encrypted data. This is, of course, not secure, but is much faster than executing in FHE. This mode is useful for debugging. This is called the Virtual Library.
- verify the maximum bitwidth of the op-graph, to determine FHE compatibility, without actually compiling the circuit to machine code. This feature is available through Concrete-Numpy and is part of the overall [FHE Assistant](../deep-learning/fhe_assistant.md).

## Bitwidth compatibility verification

The second step of compilation checks that the maximum bitwidth encountered anywhere in the circuit is valid.

If the check fails for a machine learning model, the user will need to tweak the available [quantization](quantization.md), [pruning](pruning.md) and model hyperparameters in order to obtain FHE compatibility. The Virtual Library is useful in this setting, as described in the [debugging models section.](../deep-learning/fhe_assistant.md)

## Compilation to machine code

Finally, the FHE compatible op-graph and the necessary cryptographic primitives from **Concrete-Framework** are converted to machine code.
