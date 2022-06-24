# Concrete-Numpy

**Concrete-ML** heavily depends on **Concrete-Numpy** which is our reference library for FHE computations.

Here we give a brief overview of **Concrete-Numpy** and how we use it in **Concrete-ML**. For more information on this library, we refer the reader to the proper documentation: [https://docs.zama.ai/concrete-numpy](https://docs.zama.ai/concrete-numpy)

**Concrete-Numpy** embeds some [FHE constraints](fhe_constraints.md) that we need to carefuly take into account.

In **Concrete-ML**, we use **Concrete-Numpy** in 4 differents ways:

1. Compilation of Numpy functions; once a model has been trained and is ready for inference, the user can __compile__ it. If the model respects some FHE constraints the FHE circuit is created.
   The compilation of the model allows us to get multiple information about the circuit as well as different graphs which brings us to the second point.
1. [Virtual Library](fhe_assistant.md); An important part for the user is to understand what the model looks like once compiled. Thanks to **Concrete-Numpy** we can call different API to get the information about the model. This allows the user to understand what he should do to make his model FHE compatible.
1. Execution of FHE computation; the ultimate goal is obviously to be able to run a model in FHE. Once the model is FHE friendly, trained and compiled we get a circuit (an object from `concrete.numpy.Circuit`) that we can use to do FHE inference.
1. [FHE circuit and keys serialization](client_server.md); finally, **Concrete-Numpy** offers a client server protocol over which **Concrete-ML** relies for the deployment part.

We give a simplistic example of how **Concrete-Numpy** is used in **Concrete-ML** for a simple linear regression model:

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

At this stage, we have everything we need to deploy the model using `Client` and  `Server` from `concrete.numpy`. Please refer to the [**Concrete-Numpy** implementation](https://docs.zama.ai/concrete-numpy) for more information on the deployment.

## Pre and post processing

In theory, it is possible to combine **Concrete-Numpy** with **Concrete-ML** such that the server can apply some pre or post processing before or after the execution of the model on the data. However this brings some complexity has all operations must be done in the quantized realm.

So currently there is no support for pre and post processing in FHE. Data must arrive to the FHE model already pre-processed and post-processing (if there is any) has to be done on the client machine.

We might add support for this in the future.
