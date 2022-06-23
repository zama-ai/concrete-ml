# Importing op-graphs from ONNX

Internally, **Concrete-ML** uses [ONNX](https://github.com/onnx/onnx) operators as intermediate representation (or IR) for manipulating machine learning models produced through export for [PyTorch](https://github.com/pytorch/pytorch), [Hummingbird](https://github.com/microsoft/hummingbird) and [skorch](https://github.com/skorch-dev/skorch). As ONNX is becoming the standard exchange format for neural networks, this allows **Concrete-ML** to be flexible while also making model representation manipulation quite easy. In addition, it allows for straight-forward mapping to NumPy operators, supported by **Concrete-Numpy** to use the **Concrete** stack FHE conversion capabilities.

## Steps of the conversion and compilation of a torch model to NumPy via ONNX

The diagram below gives an overview of the steps involved in the conversion of an ONNX graph to a FHE compatible format, i.e. a format that can  be compiled to FHE through **Concrete-Numpy**.

All **Concrete-ML** builtin models follow the same pattern for FHE conversion:

1. The models are trained with sklearn or torch
1. All models have a torch implementation for inference. This implementation is provided either by third-party tool such as [hummingbird](hummingbird_usage.md), or is implemented in our library
1. The torch model is exported to ONNX. For more information on the use of ONNX in **Concrete-ML** see [here](onnx.md#steps-of-the-conversion-and-compilation-of-a-torch-model-to-numpy-via-onnx)
1. Our ONNX parser checks that all the operations in the ONNX graph are supported and assigns reference numpy operations to them. This step produces a `NumpyModule`
1. Quantization is performed on the [`NumpyModule`](_apidoc/concrete.ml.torch.html#concrete.ml.torch.numpy_module.NumpyModule), producing a  [`QuantizedModule`](_apidoc/concrete.ml.quantization.html#concrete.ml.quantization.quantized_module.QuantizedModule) . Two steps are performed: calibration and assignment of equivalent [`QuantizedOp`](_apidoc/concrete.ml.quantization.html#concrete.ml.quantization.base_quantized_op.QuantizedOp)  objects to each ONNX operation. The `QuantizedModule` class is the quantized counterpart of the `NumpyModule`.
1. Once the `QuantizedModule` is built, we use **Concrete-Numpy** to trace the `._forward()` function of the `QuantizedModule`

Moreover, by passing a user provided `nn.Module` to step 2 of the above process, **Concrete-ML** supports custom user models. See the associated [custom model documentation](custom_models.md) for instructions about working with such models.

![Torch compilation flow with ONNX](./_static/compilation-pipeline/torch_to_numpy_with_onnx.svg)

Once an ONNX model is imported, it is converted to a `NumpyModule`, then to a `QuantizedModule` and, finally, to an FHE circuit. However, as the diagram shows, it is perfectly possible to stop at the `NumpyModule` level if you just want to run the torch model as NumPy code without doing quantization.

{% hint style='info' %}
Note that if you keep the obtained `NumpyModule` without quantizing it with Post Training Quantization (PTQ), it will not be convertible to FHE since the **Concrete** stack requires operators to use integers for computations.
{% endhint %}

The `NumpyModule` stores the ONNX model that it interprets. The interpreter works by going through the ONNX graph in [topological order](https://en.wikipedia.org/wiki/Topological_sorting), and storing the intermediate results as it goes. To execute a node, the interpreter feeds the required inputs - taken either from the model inputs or the intermediate results - to the NumPy implementation of each ONNX node.

## Calibration

Calibration is the process of executing the `NumpyModule` with a representative set of data, in floating point. It allows to compute statistics for all the intermediate tensors used in the network to determine quantization parameters.

{% hint style='info' %}
Note that the `NumpyModule` interpreter currently [supports the following ONNX operators](compilation_onnx.md#ops-supported-for-evaluation-numpy-conversion).
{% endhint %}

## Quantization

Quantization is the process of converting floating point weights, inputs and activations to integer, according to the quantization parameters computed during Calibration.

Initializers (model trained parameters) are quantized according to `n_bits` and passed to the Post Training Quantization (PTQ) process.

During the PTQ process, the ONNX model stored in the `NumpyModule` is interpreted and calibrated using `ONNX_OPS_TO_QUANTIZED_IMPL` dictionary, which maps ONNX operators (eg, Gemm) to their quantized equivalent (eg, QuantizedGemm). Remark that for some ONNX operator, this quantized equivalence is only partial (FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/876).

Quantized operators are then used to create a `QuantizedModule` that, similarly to the `NumpyModule`, runs through the operators to perform the quantized inference with integers-only operations.

That `QuantizedModule` is then compilable to FHE if the intermediate values conform to the 8 bits precision limit of the **Concrete** stack.

## Understanding the models with ONNX

In order to better understand how Concrete-ML works under the hood, it is possible to access each model in their ONNX format and then either either print it or visualize it by importing the associated file in [Netron](https://netron.app). For example, with `LogisticRegression`:

```python
import onnx
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression

# Create the data for classification
x, y = make_classification(n_samples=100, class_sep=2, n_features=4, random_state=42)

# Retrieve train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=10, random_state=42
)

# Fix the number of bits to used for quantization
model = LogisticRegression(n_bits=2)

# Fit the model
model.fit(X_train, y_train)

# Access to the model
onnx_model = model.onnx_model

# Print the model
print(onnx.helper.printable_graph(onnx_model.graph))

# Save the model
onnx.save(onnx_model, "tmp.onnx")

# And then visualize it with Netron
```
