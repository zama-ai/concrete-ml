# Compilation

Concrete-ML implements machine model inference using Concrete-Numpy as a backend. In order to execute in FHE, a numerical program written in Concrete-Numpy needs to be compiled. This functionality is [described here](https://docs.zama.ai/concrete-numpy/getting-started/quick_start), and Concrete-ML hides away most of the complexity of this step. The entire compilation process is done by Concrete-Numpy.

From the perspective of the Concrete-ML user, the compilation process performed by Concrete-Numpy can be broken up into 3 steps:

1. Numpy program tracing and creation of a Concrete-Numpy op-graph
1. checking that the op-graph is FHE compatible
1. producing machine code for the op-graph. This step automatically determines cryptographic parameters

Additionally, the [client/server API](client_server.md) packages the result of the last step in a way that allows to deploy the encrypted circuit to a server and to perform key generation, encryption and decryption on the client side.

## A simple Concrete-Numpy example

Here is an toy example for a simple linear regression model on integers. Note that this is just an example to illustrate compilation concepts, we recommend using the [built-in models](../built-in-models/linear.md) which provide linear regression out of the box.

```python
import numpy
from concrete.numpy.compilation import compiler

# Let's assume Quantization has been applied and we are left with integers only.
# This is essentially the work of Concrete-ML

# Some parameters (weight and bias) for our model taking a single feature
w = [2]
b = 2

# The function that implements our model
@compiler({"x": "encrypted"})
def linear_model(x):
    return w @ x + b

# A representative inputset is needed to compile the function
# (used for tracing)
n_bits_input = 2
inputset = numpy.arange(0, 2**n_bits_input).reshape(-1, 1)
circuit = linear_model.compile(inputset)

# Use the API to get the maximum bitwidth in the circuit
max_bitwidth = circuit.graph.maximum_integer_bit_width()
print("Max bitwidth = ", max_bitwidth)
# Max bitwidth =  4

# Test our FHE inference
circuit.encrypt_run_decrypt(numpy.array([3]))
# 8

# Print the graph of the circuit
print(circuit)
# %0 = 2                     # ClearScalar<uint2>
# %1 = [2]                   # ClearTensor<uint2, shape=(1,)>
# %2 = x                     # EncryptedTensor<uint2, shape=(1,)>
# %3 = matmul(%1, %2)        # EncryptedScalar<uint3>
# %4 = add(%3, %0)           # EncryptedScalar<uint4>
# return %4
```

## Concrete-Numpy op-graphs and the Virtual Library

The first step in the list above takes a python function implemented using the Concrete-Numpy [supported operation set](https://docs.zama.ai/concrete-numpy/getting-started/compatibility) and transforms it into an executable operation graph. In this step all the floating point subgraphs in the op-graph are fused and converted to Table Lookup operations.

This enables to:

- execute the op-graph, which includes TLUs, on clear non-encrypted data. This is, of course, not secure, but is much faster than executing in FHE. This mode is useful for debugging. This is called the Virtual Library.
- verify the maximum bitwidth of the op-graph, to determine FHE compatibility, without actually compiling the circuit to machine code. This feature is available through Concrete-Numpy and is part of the overall [FHE Assistant](../deep-learning/fhe_assistant.md).

## Bitwidth compatibility verification

The second step of compilation checks that the maximum bitwidth encountered anywhere in the circuit is valid.

If the check fails for a machine learning model, the user will need to tweak the available [quantization](quantization.md), [pruning](pruning.md) and model hyperparameters in order to obtain FHE compatibility. The Virtual Library is useful in this setting, as described in the [debugging models section.](../deep-learning/fhe_assistant.md)

## Compilation to machine code

Finally, the FHE compatible op-graph and the necessary cryptographic primitives from **Concrete-Framework** are converted to machine code.
