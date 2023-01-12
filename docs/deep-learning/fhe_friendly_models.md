# Step-by-Step Guide

This guide provides a complete example of converting a Pytorch neural network into its FHE-friendly quantized counterpart. It focuses on Quantization Aware Training a simple network on a synthetic data-set.

In general, quantization can be carried out in two different ways: either during training with Quantization Aware Training (QAT), or after the training phase with Post-Training Quantization (PTQ).

Regarding FHE-friendly neural networks, Quantization Aware Training is the best way to reach optimal accuracy under [FHE constrains](../README.md#current-limitations). This technique allows weights and activations to be reduced to very low bit-widths (e.g. 2-3 bits), which, combined with pruning, can keep accumulator bit-widths low.

Concrete-ML uses a third party library, [Brevitas](https://github.com/Xilinx/brevitas) to perform QAT for Pytorch NNs, but other options exist for other frameworks such as Keras/Tensorflow.

Several [demos and tutorials](../getting-started/showcase.md) that use Brevitas are available in Concrete-ML library, such as the [CIFAR classification tutorial](https://github.com/zama-ai/concrete-ml/blob/release/0.6.x/use_case_examples/cifar_brevitas_finetuning/CifarQuantizationAwareTraining.ipynb).

This guide is based on a [notebook tutorial](https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/docs/advanced_examples/QuantizationAwareTraining.ipynb), from which some code blocks are documented here.

For a more formal description of the usage of Brevitas to build FHE-compatible neural networks, please see the [Brevitas usage reference](../developer-guide/external_libraries.md#brevitas).

## Baseline Torch model

In PyTorch, using standard layers, a fully connected neural network would look as follows:

```python
import torch
from torch import nn

IN_FEAT = 2
OUT_FEAT = 2

class SimpleNet(nn.Module):
    """Simple MLP with PyTorch"""

    def __init__(self, n_hidden = 30):
        super().__init__()
        self.fc1 = nn.Linear(in_features=IN_FEAT, out_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.fc3 = nn.Linear(in_features=n_hidden, out_features=OUT_FEAT)


    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

The [notebook tutorial](https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/docs/advanced_examples/QuantizationAwareTraining.ipynb), example shows how to train a fully-connected neural network, similar to the one above, on a synthetic 2D data-set with a checkerboard grid pattern of 100 x 100 points. The data is split into 9500 training and 500 test samples.

Once trained, this Pytorch network can be imported using the [`compile_torch_model`](../developer-guide/api/concrete.ml.torch.compile.md#function-compile_torch_model) function. This function uses simple Post-Training Quantization.

The network was trained using different numbers of neurons in the hidden layers, and quantized using 3-bits weights and activations. The mean accumulator size shown below was extracted using the [Virtual Library](fhe_assistant.md) and is measured
as the mean over 10 runs of the experiment. An accumulator of 6.6 means that 4 times out of 10 the accumulator
measured was 6 bits while 6 times it was 7 bits.

| neurons               | 10     | 30     | 100    |
| --------------------- | ------ | ------ | ------ |
| fp32 accuracy         | 68.70% | 83.32% | 88.06% |
| 3-bit accuracy        | 56.44% | 55.54% | 56.50% |
| mean accumulator size | 6.6    | 6.9    | 7.4    |

This shows that the fp32 accuracy and accumulator size increases with the number of hidden neurons, while the 3-bit accuracy remains low irrespective of to the number of neurons. While all the configurations tried here were FHE-compatible (accumulator \< 16 bits), it is often preferable to have a lower accumulator size in order to speed up the inference time.

{% hint style="info" %}
The accumulator size is determined by Concrete-Numpy as being the maximum bit-width encountered anywhere in the encrypted circuit
{% endhint %}

## Quantization Aware Training:

[Quantization Aware Training](../advanced-topics/quantization.md) using [Brevitas](https://github.com/Xilinx/brevitas) is the best way to guarantee a good accuracy for Concrete-ML compatible neural networks.

Brevitas provides a quantized version of almost all Pytorch layers (`Linear` layer becomes `QuantLinear`, `ReLU` layer becomes `QuantReLU` and so one), plus some extra quantization parameters, such as :

- `bit_width`: precision quantization bits for activations
- `act_quant`: quantization protocol for the activations
- `weight_bit_width`: precision quantization bits for weights
- `weight_quant`: quantization protocol for the weights

In order to use FHE, the network must be quantized from end to end, and thanks to the Brevitas's `QuantIdentity` layer, it's possible to quantize the input by placing it at the entry point of the network. Moreover, it's also possible to combine Pytorch and Brevitas layers, provided that a `QuantIdentity` is placed after this Pytorch layer. The following table gives the replacements to be made to convert a Pytorch NN for Concrete-ML compatibility.

| Pytorch fp32 layer   | Concrete-ML model with Pytorch/Brevitas               |
| -------------------- | ----------------------------------------------------- |
| `torch.nn.Linear`    | `brevitas.quant.QuantLinear`                          |
| `torch.nn.Conv2d`    | `brevitas.quant.Conv2d`                               |
| `torch.nn.AvgPool2d` | `torch.nn.AvgPool2d` + `brevitas.quant.QuantIdentity` |
| `torch.nn.ReLU`      | `brevitas.quant.QuantReLU`                            |

Furthermore, some Torch operators (from the Torch functional API), require a `brevitas.quant.QuantIdentity` to be applied on their inputs.

| Pytorch ops that require QuantIdentity       |
| -------------------------------------------- |
| `torch.transpose`                            |
| `torch.add` (between two activation tensors) |
| `torch.reshape`                              |
| `torch.flatten`                              |

{% hint style="info" %}
The QAT import tool in Concrete-ML is a work in progress. While it has been tested with some networks built with Brevitas, it is possible to use other tools to obtain QAT networks.
{% endhint %}

For instance with Brevitas, the network above becomes :

<!--pytest-codeblocks:cont-->

```python
from brevitas import nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

N_BITS = 3
IN_FEAT = 2
OUT_FEAT = 2

class QuantSimpleNet(nn.Module):
    def __init__(
        self,
        n_hidden,
        qlinear_args={
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True
        },
        qidentity_args={"bit_width": N_BITS, "act_quant": Int8ActPerTensorFloat},
    ):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(**qidentity_args)
        self.fc1 = qnn.QuantLinear(IN_FEAT, n_hidden, **qlinear_args)
        self.relu1 = qnn.QuantReLU(bit_width=qidentity_args["bit_width"])
        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, **qlinear_args)
        self.relu2 = qnn.QuantReLU(bit_width=qidentity_args["bit_width"])
        self.fc3 = qnn.QuantLinear(n_hidden, OUT_FEAT, **qlinear_args)

        for m in self.modules():
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x       
```

{% hint style="info" %}
Note that in the network above, biases are used for linear layers but are not quantized (`"bias": True, "bias_quant": None`). The addition of the bias is an univariate operation and is fused into the activation function.
{% endhint %}

Training this network with pruning (see below) with 30 out of 100 total non-zero neurons gives good accuracy while keeping the accumulator size low.

| Non-zero neurons              | 30    |
| ----------------------------- | ----- |
| 3-bit accuracy brevitas       | 95.4% |
| 3-bit accuracy in Concrete-ML | 95.4% |
| Accumulator size              | 7     |

{% hint style="info" %}
The PyTorch QAT training loop is the same as the standard floating point training loop, but hyper-parameters such as learning rate might need to be adjusted.
{% endhint %}

{% hint style="info" %}
Quantization Aware Training is somewhat slower than normal training. QAT introduces quantization during both the forward and backward passes. The quantization process is inefficient on GPUs as its computational intensity is low with respect to data transfer time.
{% endhint %}

### Pruning using Torch

Considering that FHE only works with limited integer precision, there is a risk of overflowing in the accumulator, which will make Concrete-ML raise an error.

To understand how to overcome this limitation, consider a scenario where 2 bits are used for weights and layer inputs/outputs. The `Linear` layer computes a dot product between weights and inputs $$y = \sum_i w_i x_i$$. With 2 bits, no overflow can occur during the computation of the `Linear` layer as long the number of neurons does not exceed 14, i.e. the sum of 14 products of 2-bit numbers does not exceed 7 bits.

By default, Concrete-ML uses symmetric quantization for model weights, with values in the interval $$\left[-2^{n_{bits}-1}, 2^{n_{bits}-1}-1\right]$$. For example, for $$n_{bits}=2$$ the possible values are $$[-2, -1, 0, 1]$$, for $$n_{bits}=3$$ the values can be $$[-4,-3,-2,-1,0,1,2,3]$$.

However, in a typical setting, the weights will not all have the maximum or minimum values (e.g. $$-2^{n_{bits}-1}$$). Instead, weights typically have a normal distribution around 0, which is one of the motivating factors for their symmetric quantization. A symmetric distribution and many zero-valued weights are desirable because opposite sign weights can cancel each other out and zero weights do not increase the accumulator size.

This fact can be leveraged to train a network with more neurons, while not overflowing the accumulator, using a technique called [pruning](../advanced-topics/pruning.md), where the developer can impose a number of zero-valued weights. Torch [provides support for pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) out of the box.

The following code shows how to use pruning in the previous example:

<!--pytest-codeblocks:cont-->

```python
import torch.nn.utils.prune as prune

class PrunedQuantNet(SimpleNet):
    """Simple MLP with PyTorch"""

    pruned_layers = set()

    def prune(self, max_non_zero):
        # Linear layer weight has dimensions NumOutputs x NumInputs
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                print(name, layer)
                num_zero_weights = (layer.weight.shape[1] - max_non_zero) * layer.weight.shape[0]
                if num_zero_weights <= 0:
                    continue
                print(f"Pruning layer {name} factor {num_zero_weights}")
                prune.l1_unstructured(layer, "weight", amount=num_zero_weights)
                self.pruned_layers.add(name)

    def unprune(self):
        for name, layer in self.named_modules():
            if name in self.pruned_layers:
                prune.remove(layer, "weight")
                self.pruned_layers.remove(name)
```

Results with `PrunedQuantNet`, a pruned version of the `QuantSimpleNet` with 100 neurons on the hidden layers, are given below, showing a mean accumulator size measured over 10 runs of the experiment:

| Non-zero neurons      | 10     | 30     |
| --------------------- | ------ | ------ |
| 3-bit accuracy        | 82.50% | 88.06% |
| Mean accumulator size | 6.6    | 6.8    |

This shows that the fp32 accuracy has been improved while maintaining constant mean accumulator size.

When pruning a larger neural network during training, it is easier to obtain a low bit-width accumulator while maintaining better final accuracy. Thus, pruning is more robust than training a similar smaller network.
