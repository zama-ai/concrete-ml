# Compilation

Compilation of a model produces machine code that executes the model on encrypted data. In some cases, notably in the client/server setting, the compilation can be done by the server when loading the model for serving.

As FHE execution is much slower than execution on non-encrypted data, Concrete ML has a simulation mode which can help to quickly evaluate the impact of FHE execution on models.

## Compilation to FHE

Concrete ML implements model inference using Concrete as a backend. In order to execute in FHE, a numerical program written in Concrete needs to be compiled. This functionality is [described here](https://docs.zama.ai/concrete/get-started/quick_start), and Concrete ML hides away most of the complexity of this step, completing the entire compilation process itself.

From the perspective of the Concrete ML user, the compilation process performed by Concrete can be broken up into 3 steps:

1. tracing the NumPy program and creating a Concrete op-graph
1. checking the op-graph for FHE compatibility
1. producing machine code for the op-graph (this step automatically determines cryptographic parameters)

Additionally, the [client/server API](../guides/client_server.md) packages the result of the last step in a way that allows the deployment of the encrypted circuit to a server, as well as key generation, encryption, and decryption on the client side.

### Built-in models

Compilation is performed for built-in models with the `compile` method :

<!--pytest-codeblocks:skip-->

```python
    clf.compile(X_train)
```

### scikit-learn pipelines

When using a pipeline, the Concrete ML model can predict with FHE during the pipeline execution, but it needs to be compiled beforehand. The compile function must be called on the Concrete ML model:

```python
import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Create the data for classification:
X, y = make_classification(
    n_features=30,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
    n_samples=250,
)

# Retrieve train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model_pca = Pipeline(
    [
        ("preprocessor", PCA()),
        ("cml_model", LogisticRegression(n_bits=8))
    ]
)

model_pca.fit(X_train, y_train)

# Compile the Concrete ML model
model_pca["cml_model"].compile(X_train)

model_pca.predict(X_test[[0]], fhe="execute")
```

### Custom models

For custom models, with one of the `compile_brevitas_qat_model` (for Brevitas models with Quantization Aware Training) or `compile_torch_model` (PyTorch models using Post-Training Quantization) functions:

<!--pytest-codeblocks:skip-->

```python
    quantized_numpy_module = compile_brevitas_qat_model(torch_model, X_train)
```

## FHE simulation

The first step in the list above takes a Python function implemented using the Concrete [supported operation set](https://docs.zama.ai/concrete/getting-started/compatibility) and transforms it into an executable operation graph.

The result of this single step of the compilation pipeline allows the:

- execution of the op-graph, which includes TLUs, on clear non-encrypted data. This is not secure, but it is much faster than executing in FHE. This mode is useful for debugging, especially when looking for appropriate model hyper-parameters
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

While Concrete ML hides away all the Concrete code that performs model inference, it can be useful to understand how Concrete code works. Here is a toy example for a simple linear regression model on integers to illustrate compilation concepts. Generally, it is recommended to use the [built-in models](../built-in-models/linear.md), which provide linear regression out of the box.

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
