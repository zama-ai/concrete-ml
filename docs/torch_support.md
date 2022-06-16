# Torch support

**Concrete-ML** supports a variety of torch operators that can be used to build fully connected
or convolutional neural networks, with normalization and activation layers. Moreover, many
element-wise operators are supported.

Our torch conversion pipeline uses ONNX and an intermediate representation. We refer the user to [the Concrete-ML ONNX operator reference](onnx.md) for more information.

## List of supported torch operators

The following operators in torch will be exported as **Concrete-ML** compatible ONNX operators:

- [`torch.abs`](https://pytorch.org/docs/stable/generated/torch.abs.html)
- [`torch.clip`](https://pytorch.org/docs/stable/generated/torch.clip.html)
- [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html)
- [`torch.nn.identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html)
- [`torch.log`](https://pytorch.org/docs/stable/generated/torch.log.html)
- [`torch.reshape`](https://pytorch.org/docs/stable/generated/torch.reshape.html)
- [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view)

Operators that take an encrypted input and unencrypted constants:

- [`torch.add`, torch.Tensor operator +](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html)
- [`torch.conv2d`, `torch.nn.Conv2D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html)
- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

## List of supported activations

Note that the equivalent versions from `torch.functional` are also supported.

<!--- List done by hand from a look to the activation_function list in test_compile_torch_activations test -->

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
