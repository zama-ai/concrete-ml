# Using Torch

In addition to the built-in models, Concrete-ML supports generic machine learning models implemented with Torch, or [exported as ONNX graphs](onnx_support.md).

As [Quantization Aware Training (QAT)](../advanced-topics/quantization.md) is the most appropriate method of training
neural networks that are compatible with [FHE constraints](../getting-started/concepts.md#model-accuracy-considerations-under-fhe-constraints), Concrete-ML works with [Brevitas](../developer-guide/external_libraries.md#brevitas), a library providing QAT support for PyTorch.

The following example uses a simple QAT PyTorch model that implements a fully connected neural network with two hidden layers. Due to its small size, making this model respect FHE constraints is relatively easy.

```python
import brevitas.nn as qnn
import torch.nn as nn
import torch

N_FEAT = 12

n_bits = 3
class QATSimpleNet(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(N_FEAT, n_hidden, True, weight_bit_width=n_bits, bias_quant=None)
        self.quant2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, True, weight_bit_width=3, bias_quant=None)
        self.quant3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(n_hidden, 2, True, weight_bit_width=n_hidden, bias_quant=None)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.quant2(torch.relu(self.fc1(x)))
        x = self.quant3(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

```

Once the model is trained, calling the [`compile_brevitas_qat_model`](../developer-guide/api/concrete.ml.torch.compile.md#function-compilebrevitasqatmodel) from Concrete-ML will automatically perform conversion and compilation of a quantization aware trained network. Here, 3-bit quantization is used for both the weights and activations.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy
torch_input = torch.randn(100, N_FEAT)
torch_model = QATSimpleNet(30)
quantized_numpy_module = compile_brevitas_qat_model(
    torch_model, # our model
    torch_input, # a representative input-set to be used for both quantization and compilation
    n_bits = n_bits,
)

```

The model can now be used to perform encrypted inference. Next, the test data is quantized:

<!--pytest-codeblocks:cont-->

```python
x_test = numpy.array([numpy.random.randn(N_FEAT)])
x_test_quantized = quantized_numpy_module.quantize_input(x_test)
```

and the encrypted inference run using either:

- `quantized_numpy_module.forward_and_dequant()` to compute predictions in the clear, on quantized data and then de-quantize the result. The return value of this function contains the dequantized (float) output of running the model in the clear. Calling the forward function on the clear data is useful when debugging. The results in FHE will be the same as those on clear quantized data.
- `quantized_numpy_module.forward_fhe.encrypt_run_decrypt()` to perform the FHE inference. In this case, dequantization is done in a second stage using `quantized_numpy_module.dequantize_output()`.

## Generic quantization aware training import

While the example above shows how to import a Brevitas/PyTorch model, Concrete-ML also provides an option to import generic quantization aware trained (QAT) models implemented either in
PyTorch or through ONNX.

QAT models contain quantizers in the PyTorch graph. These quantizers ensure that the inputs to the Linear/Dense and Conv layers are quantized.

Suppose that `n_bits_qat` is the bit-width of activations and weights during the QAT process. To import a PyTorch QAT network you can use the [`compile_torch_model`](../developer-guide/api/concrete.ml.torch.compile.md#function-compiletorchmodel) library function, passing `import_qat=True`:

<!--pytest-codeblocks:skip-->

```python
from concrete.ml.torch.compile import compile_torch_model
n_bits_qat = 3

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    import_qat=True,
    n_bits=n_bits_qat,
)
```

Alternatively, if you want to import directly an ONNX model, please see [the ONNX guide](onnx_support.md). The [`compile_onnx_model`](../developer-guide/api/concrete.ml.torch.compile.md#function-compileonnxmodel) also supports the `import_qat` parameter.

{% hint style="warning" %}
When importing QAT models using this generic pipeline, a representative calibration set should be given, as quantization parameters in the model need to be inferred from the statistics of the values encountered during inference.
{% endhint %}

## Supported Operators and Activations

Concrete-ML supports a variety of PyTorch operators that can be used to build fully connected or convolutional neural networks, with normalization and activation layers. Moreover, many element-wise operators are supported.

### Operators

#### Univariate operators

- [`torch.abs`](https://pytorch.org/docs/stable/generated/torch.abs.html)
- [`torch.clip`](https://pytorch.org/docs/stable/generated/torch.clip.html)
- [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html)
- [`torch.log`](https://pytorch.org/docs/stable/generated/torch.log.html)
- [`torch.gt`](https://pytorch.org/docs/stable/generated/torch.gt.html)
- [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html)
- [`torch.mul, torch.Tensor operator *`](https://pytorch.org/docs/stable/generated/torch.mul.html)
- [`torch.div, torch.Tensor operator /`](https://pytorch.org/docs/stable/generated/torch.div.html)
- [`torch.nn.identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html)

#### Shape modifying operators

- [`torch.reshape`](https://pytorch.org/docs/stable/generated/torch.reshape.html)
- [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view)
- [`torch.flatten`](https://pytorch.org/docs/stable/generated/torch.flatten.html)
- [`torch.transpose`](https://pytorch.org/docs/stable/generated/torch.transpose.html)

#### Operators that take an encrypted input and unencrypted constants

- [`torch.conv2d`, `torch.nn.Conv2D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html)
- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

Please note that Concrete-ML supports these operators but also the Quantization Aware Training equivalents from Brevitas.

- `brevitas.nn.QuantLinear`
- `brevitas.nn.QuantConv2d`

#### Operators that can take both encrypted+unencrypted and encrypted+encrypted inputs

- [`torch.add, torch.Tensor operator +`](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html)
- [`torch.sub, torch.Tensor operator -`](https://pytorch.org/docs/stable/generated/torch.Tensor.sub.html)

### Quantizers

- `brevitas.nn.QuantIdentity`

### Activations

- [`torch.nn.Celu`](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
- [`torch.nn.Elu`](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
- [`torch.nn.GELU`](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [`torch.nn.Hardshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html)
- [`torch.nn.HardSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html)
- [`torch.nn.Hardswish`](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish)
- [`torch.nn.HardTanh`](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)
- [`torch.nn.LeakyRelu`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- [`torch.nn.LogSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html)
- [`torch.nn.Mish`](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html)
- [`torch.nn.PReLU`](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
- [`torch.nn.ReLU6`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)
- [`torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- [`torch.nn.Selu`](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html)
- [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
- [`torch.nn.SiLU`](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [`torch.nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
- [`torch.nn.Softshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html)
- [`torch.nn.Softsign`](https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html)
- [`torch.nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
- [`torch.nn.Tanhshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html)
- [`torch.nn.Threshold`](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html) -- partial support

{% hint style="info" %}
Note that the equivalent versions from `torch.functional` are also supported.
{% endhint %}
