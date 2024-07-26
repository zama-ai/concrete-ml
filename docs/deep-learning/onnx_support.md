# Using ONNX

This document explains how to compile [ONNX](https://onnx.ai/) models in Concrete ML. This is particularly useful for importing models trained with Keras.

You can compile ONNX models by directly importing models that are already quantized with [Quantization Aware Training (QAT)](../getting-started/concepts.md#i-model-development) or by performing [Post Training Quantization (PTQ)](../getting-started/concepts.md#i-model-development) with Concrete ML.

## Simple example

The following example shows how to compile an ONNX model using PTQ. The model was initially trained using Keras before being exported to ONNX. The training code is not shown here.

{% hint style="warning" %}
This example uses PTQ, meaning that the quantization is not performed during training. This model does not have the optimal performance in FHE.

To improve performance in FHE, you should add QAT. Additionally, you can also import QAT ONNX models [as shown below](onnx_support.md#quantization-aware-training).
{% endhint %}

```python
import numpy
import onnx

from concrete.ml.torch.compile import compile_onnx_model
from concrete.fhe.compilation import Configuration



n_bits = 6
input_output_feature = 5
input_shape = (input_output_feature,)
num_inputs = 1
n_examples = 5000

# Create random input
input_set = numpy.random.uniform(-100, 100, size=(n_examples, *input_shape))

onnx_model = onnx.load(f"tests/data/tf_onnx/fc_{input_output_feature}.onnx")
onnx.checker.check_model(onnx_model)

# Compile
quantized_module = compile_onnx_model(
    onnx_model, input_set, n_bits=2
)

# Create test data from the same distribution and quantize using
# learned quantization parameters during compilation
x_test = tuple(numpy.random.uniform(-100, 100, size=(1, *input_shape)) for _ in range(num_inputs))

y_clear = quantized_module.forward(*x_test, fhe="disable")
y_fhe = quantized_module.forward(*x_test, fhe="execute")

print("Execution in clear: ", y_clear)
print("Execution in FHE:   ", y_fhe)
print("Equality:           ", numpy.sum(y_clear == y_fhe), "over", numpy.size(y_fhe), "values")
```

{% hint style="warning" %}
While a Keras ONNX model was used in this example, Keras/Tensorflow support in Concrete ML is only partial and experimental.
{% endhint %}

## Quantization Aware Training

Models trained using QAT contain quantizers in the ONNX graph. These quantizers ensure that the inputs to the Linear/Dense and Conv layers are quantized. Since these QAT models have quantizers configured to a specific number of bits during training, you must import the ONNX graph using the same settings:

<!--pytest-codeblocks:skip-->

```python
# Define the number of bits to use for quantizing weights and activations during training
n_bits_qat = 3  

quantized_numpy_module = compile_onnx_model(
    onnx_model,
    input_set,
    import_qat=True,
    n_bits=n_bits_qat,
)
```

## Supported operators

Concrete ML supports the following operators for evaluation and conversion to an equivalent FHE circuit. Other operators were not implemented either due to FHE constraints or because they are rarely used in PyTorch activations or scikit-learn models.

<!--- gen_supported_ops.py: inject supported operations for evaluation [BEGIN] -->

<!--- do not edit, auto generated part by `make supported_ops` -->

- Abs
- Acos
- Acosh
- Add
- Asin
- Asinh
- Atan
- Atanh
- AveragePool
- BatchNormalization
- Cast
- Celu
- Clip
- Concat
- Constant
- ConstantOfShape
- Conv
- Cos
- Cosh
- Div
- Elu
- Equal
- Erf
- Exp
- Expand
- Flatten
- Floor
- Gather
- Gemm
- Greater
- GreaterOrEqual
- HardSigmoid
- HardSwish
- Identity
- LeakyRelu
- Less
- LessOrEqual
- Log
- MatMul
- Max
- MaxPool
- Min
- Mul
- Neg
- Not
- OneHot
- Or
- PRelu
- Pad
- Pow
- ReduceSum
- Relu
- Reshape
- Round
- Selu
- Shape
- Sigmoid
- Sign
- Sin
- Sinh
- Slice
- Softplus
- Squeeze
- Sub
- Tan
- Tanh
- ThresholdedRelu
- Transpose
- Unfold
- Unsqueeze
- Where
- onnx.brevitas.Quant

<!--- gen_supported_ops.py: inject supported operations for evaluation [END] -->
