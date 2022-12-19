# Step-by-Step Guide

This section includes a complete example of converting a neural network to Quantization Aware Training (QAT). This tutorial uses PyTorch and Brevitas to train a simple network on a synthetic data-set. You can find the demo of the final network in the [custom-model with quantization aware training demo](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/QuantizationAwareTraining.ipynb). To see how to apply these network design principles for a real-world data-set, please see the [MNIST use-case example](https://github.com/zama-ai/concrete-ml-internal/blob/main/use%5C_case%5C_examples/mnist/mnist%5C_in%5C_fhe.ipynb).

For a more formal description of the usage of Brevitas to build FHE-compatible neural networks, please see the [Brevitas usage reference](../developer-guide/external_libraries.md#brevitas).

## Summary

- [Step-by-Step Guide](#step-by-step-guide)
  - [Summary](#summary)
  - [Baseline model](#baseline-model)
    - [Pruning using Torch](#pruning-using-torch)
  - [Quantization Aware Training](#quantization-aware-training)

## Baseline model

This example shows how to train a fully-connected neural network on a synthetic 2D data-set with a checkerboard grid pattern of 100 x 100 points. The data is split into 9500 training and 500 test samples.

In PyTorch, using standard layers, this network would look as follows:

```python
from torch import nn
import torch

N_FEAT = 2
class SimpleNet(nn.Module):
    """Simple MLP with PyTorch"""

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

Once trained, this network can be imported using the [`compile_torch_model`](../developer-guide/api/concrete.ml.torch.compile.md#function-compile_torch_model) function. This function uses simple Post-Training Quantization.

The network was trained using different numbers of neurons in the hidden layers, and quantized using 3-bits weights and activations. The mean accumulator size shown below was extracted using the [Virtual Library](fhe_assistant.md).

| neurons               | 10     | 30     | 100    |
| --------------------- | ------ | ------ | ------ |
| fp32 accuracy         | 68.70% | 83.32% | 88.06% |
| 3bit accuracy         | 56.44% | 55.54% | 56.50% |
| mean accumulator size | 6.6    | 6.9    | 7.4    |

This shows that the fp32 accuracy and accumulator size increases with the number of hidden neurons, while the 3-bit accuracy remains low irrespective of to the number of neurons. While all the configurations tried here were FHE-compatible (accumulator \< 16 bits), it is often preferable to have a lower accumulator size in order to speed up the inference time.

{% hint style="info" %}
The accumulator size is determined by Concrete-Numpy as being the maximum bit-width encountered anywhere in the encrypted circuit
{% endhint %}

### Pruning using Torch

Considering that FHE only works with limited integer precision, there is a risk of overflowing in the accumulator, resulting in unpredictable results.

To understand how to overcome this limitation, consider a scenario where 2 bits are used for weights and layer inputs/outputs. The `Linear` layer computes a dot product between weights and inputs $$y = \sum_i w_i x_i$$. With 2 bits, no overflow can occur during the computation of the `Linear` layer as long the number of neurons does not exceed 14, i.e. the sum of 14 products of 2-bit numbers does not exceed 7 bits.

By default, Concrete-ML uses symmetric quantization for model weights, with values in the interval $$\left[-2^{n_{bits}-1}, 2^{n_{bits}-1}-1\right]$$. For example, for $$n_{bits}=2$$ the possible values are $$[-2, -1, 0, 1]$$, for $$n_{bits}=3$$ the values can be $$[-4,-3,-2,-1,0,1,2,3]$$.

However, in a typical setting, the weights will not all have the maximum or minimum values (e.g. $$-2^{n_{bits}-1}$$). Instead, weights typically have a normal distribution around 0, which is one of the motivating factors for their symmetric quantization. A symmetric distribution and many zero-valued weights are desirable because opposite sign weights can cancel each other out and zero weights do not increase the accumulator size.

This can be leveraged to train a network with more neurons, while not overflowing the accumulator, using a technique called [pruning](../advanced-topics/pruning.md), where the developer can impose a number of zero-valued weights. Torch [provides support for pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) out of the box.

The following code shows how to use pruning in the previous example:

<!--pytest-codeblocks:cont-->

```python
import torch.nn.utils.prune as prune

class PrunedSimpleNet(SimpleNet):
    """Simple MLP with PyTorch"""

    def prune(self, max_non_zero, enable):
        # Linear layer weight has dimensions NumOutputs x NumInputs
        for layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                num_zero_weights = (layer.weight.shape[1] - max_non_zero) * layer.weight.shape[0]
                if num_zero_weights <= 0:
                    continue

                if enable:
                    prune.l1_unstructured(layer, "weight", amount=num_zero_weights)
                else:
                    prune.remove(layer, "weight")
```

Results with `PrunedSimpleNet`, a pruned version of the `SimpleNet` with 100 neurons on the hidden layers, are given below:

| non-zero neurons      | 10     | 30     |
| --------------------- | ------ | ------ |
| fp32 accuracy         | 82.50% | 88.06% |
| 3bit accuracy         | 57.74% | 57.82% |
| mean accumulator size | 6.6    | 6.8    |

This shows that the fp32 accuracy has been improved while maintaining constant mean accumulator size.

When pruning a larger neural network during training, it is easier to obtain a low bit-width accumulator while maintaining better final accuracy. Thus, pruning is more robust than training a similar smaller network.

## Quantization Aware Training

While pruning helps maintain the post-quantization level of accuracy in low-precision settings, it does not help maintain accuracy when quantizing from floating point models. The best way to guarantee accuracy is to use QAT (read more in the [quantization documentation](../advanced-topics/quantization.md)).

In this example, QAT is done using [Brevitas](https://github.com/Xilinx/brevitas), changing `Linear` layers to `QuantLinear` and adding quantizers on the inputs of linear layers using `QuantIdentity.`

{% hint style="info" %}
The QAT import tool in Concrete-ML is a work in progress. While it has been tested with some networks built with Brevitas, it is possible to use other tools to obtain QAT networks.
{% endhint %}

<!--pytest-codeblocks:cont-->

```python
import brevitas.nn as qnn


from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from dependencies import value

# Configure quantization options
class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT

# Quantization options for weights/activations
class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0
    signed = True


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0

class QATPrunedSimpleNet(nn.Module):
    def __init__(self, n_hidden):
        super(QATPrunedSimpleNet, self).__init__()

        n_bits = 3
        self.quant_inp = qnn.QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=n_bits,
            return_quant_tensor=True,
        )

        self.fc1 = qnn.QuantLinear(
            N_FEAT,
            n_hidden,
            True,
            weight_quant=CommonWeightQuant,
            weight_bit_width=n_bits,
            bias_quant=None,
        )

        self.q1 = qnn.QuantIdentity(
            act_quant=CommonActQuant, bit_width=n_bits, return_quant_tensor=True
        )

        self.fc2 = qnn.QuantLinear(
            n_hidden,
            n_hidden,
            True,
            weight_quant=CommonWeightQuant,
            weight_bit_width=3,
            bias_quant=None
        )

        self.q2 = qnn.QuantIdentity(
            act_quant=CommonActQuant, bit_width=n_bits, return_quant_tensor=True
        )

        self.fc3 = qnn.QuantLinear(
            n_hidden,
            2,
            True,
            weight_quant=CommonWeightQuant,
            weight_bit_width=n_hidden,
            bias_quant=None,
        )

        for m in self.modules():
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.q1(torch.relu(self.fc1(x)))
        x = self.q2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def prune(self, max_non_zero, enable):
        # Linear layer weight has dimensions NumOutputs x NumInputs
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                num_zero_weights = (layer.weight.shape[1] - max_non_zero) * layer.weight.shape[0]
                if num_zero_weights <= 0:
                    continue

                if enable:
                    print(f"Pruning layer {name} factor {num_zero_weights}")
                    prune.l1_unstructured(layer, "weight", amount=num_zero_weights)
                else:
                    prune.remove(layer, "weight")
```

Training this network with 30 out of 100 total non-zero neurons gives good accuracy while keeping the accumulator size low.

| non-zero neurons             | 30    |
| ---------------------------- | ----- |
| 3bit accuracy brevitas       | 95.4% |
| 3bit accuracy in Concrete-ML | 92.4% |
| accumulator size             | 7     |

{% hint style="info" %}
The PyTorch QAT training loop is the same as the standard floating point training loop, but hyper-parameters such as learning rate might need to be adjusted.
{% endhint %}

{% hint style="info" %}
Quantization Aware Training is somewhat slower than normal training. QAT introduces quantization during both the forward and backward passes. The quantization process is inefficient on GPUs as its computational intensity is low with respect to data transfer time.
{% endhint %}
