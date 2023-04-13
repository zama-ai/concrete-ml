# Compilation

Compilation of a model produces machine code that executes the model on encrypted data. In some cases, notably in the client/server setting, the compilation can be done by the server when loading the model for serving.

As FHE execution is much slower than execution on non-encrypted data, Concrete ML has a simulation mode which can help to quickly evaluate the impact of FHE execution on models.

## Compilation to FHE

Concrete ML implements model inference using Concrete as a backend. In order to execute in FHE, a numerical program written in Concrete needs to be compiled. This functionality is [described here](https://docs.zama.ai/concrete/getting-started/quick_start), and Concrete ML hides away most of the complexity of this step, completing the entire compilation process itself.

From the perspective of the Concrete ML user, the compilation process performed by Concrete can be broken up into 3 steps:

1. tracing the NumPy program and creating a Concrete op-graph
1. checking the op-graph for FHE compatability
1. producing machine code for the op-graph (this step automatically determines cryptographic parameters)

Additionally, the [client/server API](client_server.md) packages the result of the last step in a way that allows the deployment of the encrypted circuit to a server, as well as key generation, encryption, and decryption on the client side.

Compilation is performed for built-in models with the `compile` method :

<!--pytest-codeblocks:skip-->	

```python
    clf.compile(X_train)
```

and, for custom models, with one of the `compile_brevitas_qat_model` (for Brevitas models with quantization aware training) or `compile_torch_model` (PyTorch models using post-training quantization) functions:

<!--pytest-codeblocks:skip-->	

```python
    quantized_numpy_module = compile_brevitas_qat_model(torch_model, X_train)
```

## FHE Simulation

The first step in the list above takes a Python function implemented using the Concrete [supported operation set](https://docs.zama.ai/concrete/getting-started/compatibility) and transforms it into an executable operation graph.

The result of this single step of the compilation pipeline allows the:

- execution of the op-graph, which includes TLUs, on clear non-encrypted data. This is, of course, not secure, but it is much faster than executing in FHE. This mode is useful for debugging, i.e. to find the appropriate model hyper-parameters
- verification of the maximum bit-width of the op-graph and the intermediary bit-widths of model layers, to evaluate their impact on FHE execution latency

Simulation is enabled for all Concrete ML models once they are compiled as shown above. Obtaining the simulated predictions of the models is done by setting the `fhe="simulate"` argument to prediction methods:

<!--pytest-codeblocks:skip-->	

```python
    Z = clf.predict_proba(X, fhe="simulate")
```

Moreover, the maximum accumulator bit-width is determined as follows:

<!--pytest-codeblocks:skip-->	

```python
    bit_width = clf.quantized_module_.fhe_circuit.graph.maximum_integer_bit_width()
```

## A simple Concrete example

While Concrete ML hides away all the Concrete code that performs model inference, it can be useful to understand how Concrete code works. Here is a toy example for a simple linear regression model on integers. Note that this is just an example to illustrate compilation concepts. Generally, it is recommended to use the [built-in models](../built-in-models/linear.md), which provide linear regression out of the box.

```python
import numpy
from concrete.fhe import compiler

# Assume Quantization has been applied and we are left with integers only. This is essentially the work of Concrete ML

# Some parameters (weight and bias) for our model taking a single feature
w = [2]
b = 2

# The function that implements our model
@compiler({"x": "encrypted"})
def linear_model(x):
    return w @ x + b

# A representative input-set is needed to compile the function (used for tracing)
n_bits_input = 2
inputset = numpy.arange(0, 2**n_bits_input).reshape(-1, 1)
circuit = linear_model.compile(inputset)

# Use the API to get the maximum bit-width in the circuit
max_bit_width = circuit.graph.maximum_integer_bit_width()
print("Max bit_width = ", max_bit_width)
# Max bit_width = 4

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
