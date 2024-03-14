# Using Torch

In addition to the built-in models, Concrete ML supports generic machine learning models implemented with Torch, or [exported as ONNX graphs](onnx_support.md).

As [Quantization Aware Training (QAT)](../explanations/quantization.md) is the most appropriate method of training neural networks that are compatible with [FHE constraints](../getting-started/concepts.md#model-accuracy-considerations-under-fhe-constraints), Concrete ML works with [Brevitas](../explanations/inner-workings/external_libraries.md#brevitas), a library providing QAT support for PyTorch.

The following example uses a simple QAT PyTorch model that implements a fully connected neural network with two hidden layers. Due to its small size, making this model respect FHE constraints is relatively easy.

{% hint style="info" %}
Converting neural networks to use FHE can be done with `compile_brevitas_qat_model` or with `compile_torch_model` for post-training quantization. If the model can not be converted to FHE two types of errors can be raised: (1) crypto-parameters can not be found and, (2) table look-up bit-width limit is exceeded. See the [debugging section](fhe_assistant.md#compilation-error-debugging) if you encounter these errors.
{% endhint %}

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
        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, True, weight_bit_width=n_bits, bias_quant=None)
        self.quant3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(n_hidden, 2, True, weight_bit_width=n_bits, bias_quant=None)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.quant2(torch.relu(self.fc1(x)))
        x = self.quant3(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

```

Once the model is trained, calling the [`compile_brevitas_qat_model`](../references/api/concrete.ml.torch.compile.md#function-compile_brevitas_qat_model) from Concrete ML will automatically perform conversion and compilation of a QAT network. Here, 3-bit quantization is used for both the weights and activations. The `compile_brevitas_qat_model` function automatically identifies the number of quantization bits used in the Brevitas model.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy

torch_input = torch.randn(100, N_FEAT)
torch_model = QATSimpleNet(30)
quantized_module = compile_brevitas_qat_model(
    torch_model, # our model
    torch_input, # a representative input-set to be used for both quantization and compilation
)

```

## Configuring quantization parameters

The PyTorch/Brevitas models, created following the example above, require the user to configure quantization parameters such as `bit_width` (activation bit-width) and `weight_bit_width`. The quantization parameters, along with the number of neurons on each layer, will determine the accumulator bit-width of the network. Larger accumulator bit-widths result in higher accuracy but slower FHE inference time.

The following configurations were determined through experimentation for convolutional and dense layers.

| target accumulator bit-width | activation bit-width | weight bit-width | number of active neurons |
| ---------------------------- | -------------------- | ---------------- | ------------------------ |
| 8                            | 3                    | 3                | 80                       |
| 10                           | 4                    | 3                | 90                       |
| 12                           | 5                    | 5                | 110                      |
| 14                           | 6                    | 6                | 110                      |
| 16                           | 7                    | 6                | 120                      |

Using the templates above, the probability of obtaining the target accumulator bit-width, for a single layer, was determined experimentally by training 10 models for each of the following data-sets.

| <p>probability of obtaining<br>the accumulator bit-width</p> | 8   | 10   | 12  | 14  | 16   |
| ------------------------------------------------------------ | --- | ---- | --- | --- | ---- |
| mnist,fashion                                                | 72% | 100% | 72% | 85% | 100% |
| cifar10                                                      | 88% | 88%  | 75% | 75% | 88%  |
| cifar100                                                     | 73% | 88%  | 61% | 66% | 100% |

Note that the accuracy on larger data-sets, when the accumulator size is low, is also reduced strongly.

| <p>accuracy for target<br>accumulator bit-width</p> | 8   | 10  | 12  | 14  | 16  |
| --------------------------------------------------- | --- | --- | --- | --- | --- |
| cifar10                                             | 20% | 37% | 89% | 90% | 90% |
| cifar100                                            | 6%  | 30% | 67% | 69% | 69% |

## Running encrypted inference

The model can now perform encrypted inference.

<!--pytest-codeblocks:cont-->

```python
x_test = numpy.array([numpy.random.randn(N_FEAT)])

y_pred = quantized_module.forward(x_test, fhe="execute")
```

In this example, the input values `x_test` and the predicted values `y_pred` are floating points. The quantization (resp. de-quantization) step is done in the clear within the `forward` method, before (resp. after) any FHE computations.

## Simulated FHE Inference in the clear

The user can also perform the inference on clear data. Two approaches exist:

- `quantized_module.forward(quantized_x, fhe="simulate")`: simulates FHE execution taking into account Table Lookup errors.\
  De-quantization must be done in a second step as for actual FHE execution. Simulation takes into account the `p_error`/`global_p_error` parameters
- `quantized_module.forward(quantized_x, fhe="disable")`: computes predictions in the clear on quantized data, and then de-quantize the result. The return value of this function contains the de-quantized (float) output of running the model in the clear. Calling this function on clear data is useful when debugging, but this does not perform actual FHE simulation.

{% hint style="info" %}
FHE simulation allows to measure the impact of the Table Lookup error on the model accuracy. The Table Lookup error can be adjusted using `p_error`/`global_p_error`, as described in the [approximate computation ](../explanations/advanced_features.md#approximate-computations)section.
{% endhint %}

## Generic Quantization Aware Training import

While the example above shows how to import a Brevitas/PyTorch model, Concrete ML also provides an option to import generic QAT models implemented in PyTorch or through ONNX. Deep learning models made with TensorFlow or Keras should be usable by preliminary converting them to ONNX.

QAT models contain quantizers in the PyTorch graph. These quantizers ensure that the inputs to the Linear/Dense and Conv layers are quantized.

Suppose that `n_bits_qat` is the bit-width of activations and weights during the QAT process. To import a PyTorch QAT network, you can use the [`compile_torch_model`](../references/api/concrete.ml.torch.compile.md#function-compile_torch_model) library function, passing `import_qat=True`:

<!--pytest-codeblocks:skip-->

```python
from concrete.ml.torch.compile import compile_torch_model
n_bits_qat = 3

quantized_module = compile_torch_model(
    torch_model,
    torch_input,
    import_qat=True,
    n_bits=n_bits_qat,
)
```

Alternatively, if you want to import an ONNX model directly, please see [the ONNX guide](onnx_support.md). The [`compile_onnx_model`](../references/api/concrete.ml.torch.compile.md#function-compile_onnx_model) also supports the `import_qat` parameter.

{% hint style="warning" %}
When importing QAT models using this generic pipeline, a representative calibration set should be given as quantization parameters in the model need to be inferred from the statistics of the values encountered during inference.
{% endhint %}

## Supported operators and activations

Concrete ML supports a variety of PyTorch operators that can be used to build fully connected or convolutional neural networks, with normalization and activation layers. Moreover, many element-wise operators are supported.

### Operators

#### Univariate operators

- [`torch.nn.identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html)
- [`torch.clip`](https://pytorch.org/docs/stable/generated/torch.clip.html)
- [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html)
- [`torch.round`](https://pytorch.org/docs/stable/generated/torch.round.html)
- [`torch.floor`](https://pytorch.org/docs/stable/generated/torch.floor.html)
- [`torch.min`](https://pytorch.org/docs/stable/generated/torch.min.html)
- [`torch.max`](https://pytorch.org/docs/stable/generated/torch.max.html)
- [`torch.abs`](https://pytorch.org/docs/stable/generated/torch.abs.html)
- [`torch.neg`](https://pytorch.org/docs/stable/generated/torch.neg.html)
- [`torch.sign`](https://pytorch.org/docs/stable/generated/torch.sign.html)
- [`torch.logical_or, torch.Tensor operator ||`](https://pytorch.org/docs/stable/generated/torch.logical_or.html)
- [`torch.logical_not`](https://pytorch.org/docs/stable/generated/torch.logical_not.html)
- [`torch.gt, torch.greater`](https://pytorch.org/docs/stable/generated/torch.gt.html)
- [`torch.ge, torch.greater_equal`](https://pytorch.org/docs/stable/generated/torch.ge.html)
- [`torch.lt, torch.less`](https://pytorch.org/docs/stable/generated/torch.lt.html)
- [`torch.le, torch.less_equal`](https://pytorch.org/docs/stable/generated/torch.le.html)
- [`torch.eq`](https://pytorch.org/docs/stable/generated/torch.eq.html)
- [`torch.where`](https://pytorch.org/docs/stable/generated/torch.where.html)
- [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html)
- [`torch.log`](https://pytorch.org/docs/stable/generated/torch.log.html)
- [`torch.pow`](https://pytorch.org/docs/stable/generated/torch.pow.html)
- [`torch.sum`](https://pytorch.org/docs/stable/generated/torch.sum.html)
- [`torch.mul, torch.Tensor operator *`](https://pytorch.org/docs/stable/generated/torch.mul.html)
- [`torch.div, torch.Tensor operator /`](https://pytorch.org/docs/stable/generated/torch.div.html)
- [`torch.nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [`torch.nn.BatchNorm3d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html)
- [`torch.erf, torch.special.erf`](https://pytorch.org/docs/stable/special.html#torch.special.erf)
- [`torch.nn.functional.pad`](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html)

#### Shape modifying operators

- [`torch.reshape`](https://pytorch.org/docs/stable/generated/torch.reshape.html)
- [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view)
- [`torch.flatten`](https://pytorch.org/docs/stable/generated/torch.flatten.html)
- [`torch.unsqueeze`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)
- [`torch.squeeze`](https://pytorch.org/docs/stable/generated/torch.squeeze.html)
- [`torch.transpose`](https://pytorch.org/docs/stable/generated/torch.transpose.html)
- [`torch.concat, torch.cat`](https://pytorch.org/docs/stable/generated/torch.cat.html)
- [`torch.nn.Unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)

#### Tensor operators

- [`torch.Tensor.expand`](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html)
- [`torch.Tensor.to`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) -- for casting to dtype

#### Multi-variate operators: encrypted input and unencrypted constants

- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [`torch.conv1d`, `torch.nn.Conv1D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [`torch.conv2d`, `torch.nn.Conv2D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [`torch.nn.AvgPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
- [`torch.nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

Concrete ML also supports some of their QAT equivalents from Brevitas.

- `brevitas.nn.QuantLinear`
- `brevitas.nn.QuantConv1d`
- `brevitas.nn.QuantConv2d`

#### Multi-variate operators: encrypted+unencrypted or encrypted+encrypted inputs

- [`torch.add, torch.Tensor operator +`](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html)
- [`torch.sub, torch.Tensor operator -`](https://pytorch.org/docs/stable/generated/torch.Tensor.sub.html)
- [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html)

### Quantizers

- `brevitas.nn.QuantIdentity`

### Activation functions

- [`torch.nn.CELU`](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
- [`torch.nn.ELU`](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
- [`torch.nn.GELU`](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [`torch.nn.Hardshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html)
- [`torch.nn.HardSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html)
- [`torch.nn.Hardswish`](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish)
- [`torch.nn.HardTanh`](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)
- [`torch.nn.LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- [`torch.nn.LogSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html)
- [`torch.nn.Mish`](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html)
- [`torch.nn.PReLU`](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
- [`torch.nn.ReLU6`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)
- [`torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- [`torch.nn.SELU`](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html)
- [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
- [`torch.nn.SiLU`](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [`torch.nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
- [`torch.nn.Softshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html)
- [`torch.nn.Softsign`](https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html)
- [`torch.nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
- [`torch.nn.Tanhshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html)
- [`torch.nn.Threshold`](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html) -- partial support

{% hint style="info" %}
The equivalent versions from `torch.functional` are also supported.
{% endhint %}
