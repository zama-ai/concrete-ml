# Use **Concrete ML** with Torch

**Concrete ML** allows you to compile a torch model to its FHE counterpart.

This process executes most of the concepts described in the documentation on [how to use quantization](../../dev/explanation/use_quantization.md) and triggers the compilation to be able to run the model over homomorphically encrypted data.

```python
from torch import nn
import torch
class LogisticRegression(nn.Module):
    """LogisticRegression with torch"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=14, out_features=1)
        self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.sigmoid1(out)
        return out

torch_model = LogisticRegression()
```

```{warning}
Note that the architecture of the neural network passed to be compiled must respect some hard constraints given by FHE. Please read the our [detailed documentation](../howto/reduce_needed_precision.md) on these limitations.
```

Once your model is trained you can simply call the `compile_torch_model` function to execute the compilation.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.torch.compile import compile_torch_model
import numpy
torch_input = torch.randn(100, 14)
quantized_numpy_module = compile_torch_model(
    torch_model, # our model
    torch_input, # a representative inputset to be used for both quantization and compilation
    n_bits = 2,
)
```

You can then call `quantized_numpy_module.forward_fhe.run()` to have the FHE inference.

Now your model is ready to infer in FHE settings.

<!--pytest-codeblocks:cont-->

```python
enc_x = numpy.array([numpy.random.randn(14)])
# An example that is going to be encrypted, and used for homomorphic inference.
enc_x_q = quantized_numpy_module.quantize_input(enc_x)
fhe_prediction = quantized_numpy_module.forward_fhe.run(enc_x)
```

`fhe_prediction` contains the clear quantized output. The user can now dequantize the output to get the actual floating point prediction as follows:

<!--pytest-codeblocks:cont-->

```python
clear_output = quantized_numpy_module.dequantize_output(
    numpy.array(fhe_prediction, dtype=numpy.float32)
)
```

If you want to see more compilation examples, you can check out the [Fully Connected Neural Network](../advanced_examples/FullyConnectedNeuralNetwork.ipynb)

## List of supported torch operators

Our torch conversion pipeline uses ONNX and an intermediate representation. We refer the user to [the Concrete ML ONNX operator reference](../../user/howto/onnx_supported_ops.md) for more information.

The following operators in torch will be exported as **Concrete ML** compatible ONNX operators:

- [`torch.abs`](https://pytorch.org/docs/stable/generated/torch.abs.html)
- [`torch.clip`](https://pytorch.org/docs/stable/generated/torch.clip.html)
- [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html)
- [`torch.nn.identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html)
- [`torch.log`](https://pytorch.org/docs/stable/generated/torch.log.html)
- [`torch.reshape`](https://pytorch.org/docs/stable/generated/torch.reshape.html)
- [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view)

Operators that take an encrypted input and un-encrypted constants:

- [`torch.add`, torch.Tensor operator +](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html)
- [`torch.conv2d`, `torch.nn.Conv2D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html)
- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

## List of supported activations

Note that the equivalent versions from `torch.functional` are also supported.

- [`torch.nn.Celu`](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
- [`torch.nn.Elu`](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
- [`torch.nn.HardSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html)
- [`torch.nn.LeakyRelu`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- [`torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- [`torch.nn.ReLU6`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)
- [`torch.nn.Selu`](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html)
- [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
- [`torch.nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
- [`torch.nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
- [`torch.nn.HardTanh`](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)
