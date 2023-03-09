# Compilation

Compilation of a model produces machine code that executes the model on encrypted data. In some cases, notably in the client/server setting, the compilation can be done by the server when loading the model for serving.

As FHE execution is much slower than execution on non-encrypted data, Concrete-ML has a simulation mode, using an execution mode named the _Virtual Library_. Since, by default, the cryptographic parameters are chosen such that the results obtained in FHE are the same as those on clear data, the Virtual Library allows you to benchmark models quickly during development.

## Compilation

Concrete-ML implements machine model inference using Concrete-Numpy as a backend. In order to execute in FHE, a numerical program written in Concrete-Numpy needs to be compiled. This functionality is [described here](https://docs.zama.ai/concrete-numpy/getting-started/quick_start), and Concrete-ML hides away most of the complexity of this step, completing the entire compilation process itself.

From the perspective of the Concrete-ML user, the compilation process performed by Concrete-Numpy can be broken up into 3 steps:

1. tracing the Numpy program and creating a Concrete-Numpy op-graph
1. checking the op-graph for FHE compatability
1. producing machine code for the op-graph (this step automatically determines cryptographic parameters)

Additionally, the [client/server API](client_server.md) packages the result of the last step in a way that allows the deployment of the encrypted circuit to a server, as well as key generation, encryption, and decryption on the client side.

## Simulation with the Virtual Library

The first step in the list above takes a Python function implemented using the Concrete-Numpy [supported operation set](https://docs.zama.ai/concrete-numpy/getting-started/compatibility) and transforms it into an executable operation graph.

The result of this single step of the compilation pipeline allows the:

- execution of the op-graph, which includes TLUs, on clear non-encrypted data. This is, of course, not secure, but it is much faster than executing in FHE. This mode is useful for debugging, i.e. to find the appropriate hyper-parameters. This mode is called the Virtual Library (which is referred as [Virtual Circuits](https://docs.zama.ai/concrete-numpy/tutorials/virtual_circuits) in Concrete-Numpy).
- verification of the maximum bit-width of the op-graph, to determine FHE compatibility, without actually compiling the circuit to machine code.

Enabling Virtual Library execution requires the definition of a compilation `Configuration`. As simulation does not execute in FHE, this can be considered _unsafe_:

<!--pytest-codeblocks:skip-->	

```python
    COMPIL_CONFIG_VL = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,  # This is for our tests in Virtual Library only
    )
```

Next, the following code uses the simulation mode for built-in models:

<!--pytest-codeblocks:skip-->	

```python
    clf.compile(
        X_train,
        use_virtual_lib=True,
        configuration=COMPIL_CONFIG_VL,
    )
```

And finally, for custom models, it is possible to enable simulation using the following syntax:

<!--pytest-codeblocks:skip-->	

```python
    quantized_numpy_module = compile_torch_model(
        torch_model,  # our model
        X_train,  # a representative input-set to be used for both quantization and compilation
        n_bits={"net_inputs": 5, "op_inputs": 3, "op_weights": 3, "net_outputs": 5},
        import_qat=is_qat,  # signal to the conversion function whether the network is QAT
        use_virtual_lib=True,
        configuration=COMPIL_CONFIG_VL,
    )
```

Obtaining the simulated predictions of the models using the Virtual Library has the same syntax as execution in FHE:

<!--pytest-codeblocks:skip-->	

```python
    Z = clf.predict_proba(X, execute_in_fhe=True)
```

Moreover, the maximum accumulator bit-width is determined as follows:

<!--pytest-codeblocks:skip-->	

```python
    bit_width = clf.quantized_module_.forward_fhe.graph.maximum_integer_bit_width()
```

## A simple Concrete-Numpy example

While Concrete-ML hides away all the Concrete-Numpy code that performs model inference, it can be useful to understand how Concrete-Numpy code works. Here is a toy example for a simple linear regression model on integers. Note that this is just an example to illustrate compilation concepts. Generally, it is recommended to use the [built-in models](../built-in-models/linear.md), which provide linear regression out of the box.

```python
import numpy
from concrete.numpy import compiler

# Let's assume Quantization has been applied and we are left with integers only.
# This is essentially the work of Concrete-ML

# Some parameters (weight and bias) for our model taking a single feature
w = [2]
b = 2

# The function that implements our model
@compiler({"x": "encrypted"})
def linear_model(x):
    return w @ x + b

# A representative input-set is needed to compile the function
# (used for tracing)
n_bits_input = 2
inputset = numpy.arange(0, 2**n_bits_input).reshape(-1, 1)
circuit = linear_model.compile(inputset)

# Use the API to get the maximum bit-width in the circuit
max_bit_width = circuit.graph.maximum_integer_bit_width()
print("Max bit_width = ", max_bit_width)
# Max bit_width =  4

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
