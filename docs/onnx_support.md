# ONNX Support

In addition to **Concrete-ML** models and to [custom models in torch](torch_support.md), it is also possible to directly compile ONNX models. This can be particularly appealing, notably to import models trained with Keras (see [in this subsection](onnx_support.md#post-training-quantization). It can also be interesting in the context of QAT (see [in this subsection](onnx_support.md#importing-an-already-trained-model-with-quantized-aware-training)), since lot of ONNX are available on the web.

## ONNX models

ONNX is a standard and open-source format, and is used in lot of projects. We refer the interested reader to [ONNX site](https://onnx.ai) for more information, including links with other data science projets.

In **Concrete-ML**, we can compile ONNX models, either by performing post-training quantization (PTQ) or by directly importing models that are already quantized with quantization aware learning (QAT).

### Post training quantization

Let us show how compilation of an ONNX model in combination with our PTQ transformation, is equivalent to compilation with torch. For the sake of our example, we'll use a `Keras` model that we convert to ONNX, which we finally compile. For brevity, we do not show the training the model.

```python
import numpy
import onnx
import tensorflow
import tf2onnx

from concrete.ml.torch.compile import compile_onnx_model
from concrete.numpy.compilation import Configuration


class FC(tensorflow.keras.Model):
    """A fully-connected model."""

    def __init__(self):
        super().__init__()
        hidden_layer_size = 10
        output_size = 5

        self.dense1 = tensorflow.keras.layers.Dense(
            hidden_layer_size,
            activation=tensorflow.nn.relu,
        )
        self.dense2 = tensorflow.keras.layers.Dense(output_size, activation=tensorflow.nn.relu6)
        self.flatten = tensorflow.keras.layers.Flatten()

    def call(self, inputs):
        """Forward function."""
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.flatten(x)


n_bits = 6
input_output_feature = 2
input_shape = (input_output_feature,)
num_inputs = 1
n_examples = 5000

# Define the Keras model
keras_model = FC()
keras_model.build((None,) + input_shape)
keras_model.compute_output_shape(input_shape=(None, input_output_feature))

# Create random input
inputset = numpy.random.uniform(-100, 100, size=(n_examples, *input_shape))

# Convert to ONNX
tf2onnx.convert.from_keras(keras_model, opset=14, output_path="tmp.model.onnx")

onnx_model = onnx.load("tmp.model.onnx")
onnx.checker.check_model(onnx_model)

# Compile
quantized_numpy_module = compile_onnx_model(
    onnx_model, inputset, n_bits=2
)

# Create test data from the same distribution and quantize using
# learned quantization parameters during compilation
x_test = tuple(numpy.random.uniform(-100, 100, size=(1, *input_shape)) for _ in range(num_inputs))
qtest = quantized_numpy_module.quantize_input(x_test)

y_clear = quantized_numpy_module(*qtest)
y_fhe = quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*qtest)

print("Execution in clear: ", y_clear)
print("Execution in FHE:   ", y_fhe)
print("Equality:           ", numpy.sum(y_clear == y_fhe), "over", numpy.size(y_fhe), "values")
```

Remark that this example shows that a data scientist may use Keras with **Concrete-ML** instead of torch, if it feels more comfortable to do so. However, Keras is not officially supported. We have not yet tested all Keras' types of layer or model and we therefore are not able to guarantee that all are properly converted in ONNX equivalent objects.

### Importing an already trained model (with quantized-aware training)

While the example above shows how to import a floating point model for post-training quantization,
**Concrete-ML** also provides an option to import quantization aware trained (QAT) models.

QAT models contain quantizers in the ONNX graph. These quantizers ensure that the inputs
to the Linear/Dense and Conv layers are quantized. Furthermore, since these QAT models
have quantizers that are configured during training to a specific number of bits,
we need to ensure that we import the ONNX graph with the same setting:

<!--pytest-codeblocks:cont-->

```python
n_bits_qat = 3  # number of bits for weights and activations during training

quantized_numpy_module = compile_onnx_model(
    onnx_model,
    inputset,
    import_qat=True,
    n_bits=n_bits_qat,
)
```

## Ops supported for evaluation/NumPy conversion

The following operators have some support for evaluation and conversion to an equivalent NumPy circuit.
Do note that all operators may not be fully supported for conversion to a circuit executable in FHE. We sometimes implement only partially the operators, either because of some limits due to FHE or because we did not need more than special case for supporting e.g. PyTorch activations or scikit-learn models.

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
- Constant
- Conv
- Cos
- Cosh
- Div
- Elu
- Equal
- Erf
- Exp
- Flatten
- Gemm
- Greater
- HardSigmoid
- HardSwish
- Identity
- LeakyRelu
- Less
- Log
- MatMul
- Mul
- Not
- Or
- PRelu
- Pad
- Pow
- ReduceSum
- Relu
- Reshape
- Round
- Selu
- Sigmoid
- Sin
- Sinh
- Softplus
- Sub
- Tan
- Tanh
- ThresholdedRelu
- Transpose
- Where

<!--- gen_supported_ops.py: inject supported operations for evaluation [END] -->
