# Using Torch

In addition to the built-in models, Concrete-ML supports generic machine learning models implemented with Torch, or [exported as ONNX graphs](onnx_support.md).

The following example uses a simple torch model that implements a fully connected neural network with two hidden units. Due to its small size, making this model respect FHE constraints is relatively easy.

```python
from torch import nn
import torch

N_FEAT = 2
class SimpleNet(nn.Module):
    """Simple MLP with torch"""

    def __init__(self, n_hidden=30):
        super().__init__()
        self.fc1 = nn.Linear(in_features=N_FEAT, out_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.fc3 = nn.Linear(in_features=n_hidden, out_features=2)


    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Once the model is trained, calling the `compile_torch_model` from Concrete-ML will automatically perform post-training quantization and compilation to FHE. Here, we use a 3-bit quantization for both the weights and activations.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.torch.compile import compile_torch_model
import numpy
torch_input = torch.randn(100, N_FEAT)
torch_model = SimpleNet(30)
quantized_numpy_module = compile_torch_model(
    torch_model, # our model
    torch_input, # a representative inputset to be used for both quantization and compilation
    n_bits = 3,
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
- `quantized_numpy_module.forward_fhe.encrypt_run_decrypt()` to perform the FHE inference. In this case, de-quantization is done in a second stage using `quantized_numpy_module.dequantize_output()`.

## Quantization aware training

While the example above shows how to import a floating point model for post-training quantization, Concrete-ML also provides an option to import quantization aware trained (QAT) models.

QAT models contain quantizers in the torch graph. These quantizers ensure that the inputs to the Linear/Dense and Conv layers are quantized. Torch quantizers are not included in Concrete-ML, so you can either implement your own or use a 3rd party library such as [brevitas](https://github.com/Xilinx/brevitas) as shown in the [FHE-friendly models documentation](fhe_friendly_models.md). Custom models can have a more generic architecture and training procedure than the Concrete-ML built-in models.

Suppose that `n_bits_qat` is the bitwidth of activations and weights during the QAT process. To import a torch QAT network you can use the following library function:

<!--pytest-codeblocks:cont-->

```python
n_bits_qat = 3

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    import_qat=True,
    n_bits=n_bits_qat,
)
```

## Supported Operators and Activations

Concrete-ML supports a variety of torch operators that can be used to build fully connected or convolutional neural networks, with normalization and activation layers. Moreover, many element-wise operators are supported.

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

#### Operators that can take both encrypted+unencrypted and encrypted+encrypted inputs

- [`torch.add, torch.Tensor operator +`](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html)
- [`torch.sub, torch.Tensor operator -`](https://pytorch.org/docs/stable/generated/torch.Tensor.sub.html)

### Activations

- [`torch.nn.Celu`](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
- [`torch.nn.Elu`](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
- [`torch.nn.GELU`](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) -- sometimes accuracy issues
- [`torch.nn.Hardshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html)
- [`torch.nn.HardSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html)
- [`torch.nn.Hardswish`](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish)
- [`torch.nn.HardTanh`](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)
- [`torch.nn.LeakyRelu`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- [`torch.nn.LogSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html) -- sometimes accuracy issues
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


