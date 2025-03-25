# Step-by-step guide

This guide demonstrates how to convert a PyTorch neural network into a Fully Homomorphic Encryption (FHE)-friendly, quantized version. It focuses on Quantization Aware Training (QAT) using a simple network on a synthetic data-set. This guide is based on a [notebook tutorial](../advanced_examples/QuantizationAwareTraining.ipynb), from which some code blocks are documented.

## Quantization

In general, quantization can be carried out in two different ways:

- During the training phase with [Quantization Aware Training (QAT)](../getting-started/concepts.md#i-model-development)
- After the training phase with [Post Training Quantization (PTQ)](../getting-started/concepts.md#i-model-development).

For FHE-friendly neural networks, QAT is the best method to achieve optimal accuracy under [FHE constraints](../getting-started/README.md#current-limitations). This technique reduces weights and activations to very low bit-widths (for example, 2-3 bits). When combined with pruning, QAT helps keep low accumulator bit-widths.

Concrete ML uses the third-party library [Brevitas](https://github.com/Xilinx/brevitas) to perform QAT for PyTorch neural networks, but options exist for other frameworks such as Keras/Tensorflow. Concrete ML provides several [demos and tutorials](../tutorials/showcase.md) that use Brevitas , including the [CIFAR classification tutorial](../../use_case_examples/cifar/cifar_brevitas_finetuning/CifarQuantizationAwareTraining.ipynb). For a more formal description of the usage of Brevitas to build FHE-compatible neural networks, please see the [Brevitas usage reference](../explanations/inner-workings/external_libraries.md#brevitas).

{% hint style="info" %}
For a formal explanation of the mechanisms that enable FHE-compatible neural networks, please see the the following paper.

[Deep Neural Networks for Encrypted Inference with TFHE, 7th International Symposium, CSCML 2023](https://arxiv.org/abs/2302.10906)
{% endhint %}

## Baseline PyTorch model

In PyTorch, using standard layers, a Fully Connected Neural Network (FCNN) would look like this:

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

Similarly to the one above, the [notebook tutorial](../advanced_examples/QuantizationAwareTraining.ipynb) shows how to train a FCNN on a synthetic 2D data-set with a checkerboard grid pattern of 100 x 100 points. The data is split into 9500 training and 500 test samples.

Once trained, you can import this PyTorch network using the [`compile_torch_model`](../references/api/concrete.ml.torch.compile.md#function-compile_torch_model) function, which uses simple PTQ.

The network was trained using different numbers of neurons in the hidden layers, and quantized using 3-bits weights and activations. The mean accumulator size, shown below, is measured as the mean over 10 runs of the experiment. An accumulator size of 6.6 means that 4 times out of 10, the accumulator was 6 bits, while 6 times it was 7 bits.

| neurons               | 10     | 30     | 100    |
| --------------------- | ------ | ------ | ------ |
| fp32 accuracy         | 68.70% | 83.32% | 88.06% |
| 3-bit accuracy        | 56.44% | 55.54% | 56.50% |
| mean accumulator size | 6.6    | 6.9    | 7.4    |

This shows that the fp32 accuracy and accumulator size increases with the number of hidden neurons, while the 3-bits accuracy remains low regardless of the number of neurons. Although all configurations tested were FHE-compatible (accumulator \< 16 bits), it is often preferable to have a lower accumulator size to speed up inference time.

{% hint style="info" %}
Accumulator size is determined by [Concrete](https://docs.zama.ai/concrete) as the maximum bit-width encountered anywhere in the encrypted circuit.
{% endhint %}

## Quantization Aware Training (QAT)

Using [QAT](../explanations/quantization.md) with [Brevitas](https://github.com/Xilinx/brevitas) is the best way to guarantee a good accuracy for Concrete ML compatible neural networks.

Brevitas provides quantized versions of almost all PyTorch layers. For example, `Linear` layer becomes `QuantLinear`, and `ReLU` layer becomes `QuantReLU`. Brevitas also offers additional quantization parameters, such as:

- `bit_width`: precision quantization bits for activations
- `act_quant`: quantization protocol for the activations
- `weight_bit_width`: precision quantization bits for weights
- `weight_quant`: quantization protocol for the weights

To use FHE, the network must be quantized from end to end. With the Brevitas `QuantIdentity` layer, you can quantize the input by placing it at the network's entry point. Moreover, you can combine PyTorch and Brevitas layers, as long as a `QuantIdentity` layer follows the PyTorch layer. The following table lists the replacements needed to convert a PyTorch neural network for Concrete ML compatibility.

| PyTorch fp32 layer   | Concrete ML model with PyTorch/Brevitas               |
| -------------------- | ----------------------------------------------------- |
| `torch.nn.Linear`    | `brevitas.quant.QuantLinear`                          |
| `torch.nn.Conv2d`    | `brevitas.quant.Conv2d`                               |
| `torch.nn.AvgPool2d` | `torch.nn.AvgPool2d` + `brevitas.quant.QuantIdentity` |
| `torch.nn.ReLU`      | `brevitas.quant.QuantReLU`                            |

Some PyTorch operators (from the PyTorch functional API), require a `brevitas.quant.QuantIdentity` to be applied on their inputs.

| PyTorch ops that require QuantIdentity       |
| -------------------------------------------- |
| `torch.transpose`                            |
| `torch.add` (between two activation tensors) |
| `torch.reshape`                              |
| `torch.flatten`                              |

{% hint style="info" %}
The QAT import tool in Concrete ML is a work in progress. While it has been tested with some networks built with Brevitas, it is possible to use other tools to obtain QAT networks.
{% endhint %}

With Brevitas, the network above becomes:

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
In the network above, biases are used for linear layers but are not quantized (`"bias": True, "bias_quant": None`). The addition of the bias is a univariate operation and is fused into the activation function.
{% endhint %}

Training this network with pruning (see [below](#pruning-using-torch)) using 30 out of 100 total non-zero neurons gives good accuracy while keeping the accumulator size low.

| Non-zero neurons              | 30    |
| ----------------------------- | ----- |
| 3-bit accuracy brevitas       | 95.4% |
| 3-bit accuracy in Concrete ML | 95.4% |
| Accumulator size              | 7     |

{% hint style="info" %}
The PyTorch QAT training loop is the same as the standard floating point training loop, but hyper-parameters such as learning rate might need to be adjusted.
{% endhint %}

{% hint style="info" %}
QAT is somewhat slower than normal training. QAT introduces quantization during both the forward and backward passes. The quantization process is inefficient on GPUs due to its low computational intensity is low relative to data transfer time.
{% endhint %}

### Pruning using Torch

Considering that FHE only works with limited integer precision, there is a risk of overflowing in the accumulator, which will make Concrete ML raise an error.

To understand how to overcome this limitation, consider a scenario where 2 bits are used for weights and layer inputs/outputs. The `Linear` layer computes a dot product between weights and inputs $$y = \sum_i w_i x_i$$. With 2 bits, no overflow can occur during the computation of the `Linear` layer as long the number of neurons does not exceed 14, as in the sum of 14 products of 2-bits numbers does not exceed 7 bits.

By default, Concrete ML uses symmetric quantization for model weights, with values in the interval $$\left[-2^{n_{bits}-1}, 2^{n_{bits}-1}-1\right]$$. For example, for $$n_{bits}=2$$ the possible values are $$[-2, -1, 0, 1]$$; for $$n_{bits}=3$$, the values can be $$[-4,-3,-2,-1,0,1,2,3]$$.

In a typical setting, the weights will not all have the maximum or minimum values (such as $$-2^{n_{bits}-1}$$). Weights typically have a normal distribution around 0, which is one of the motivating factors for their symmetric quantization. A symmetric distribution and many zero-valued weights are desirable because opposite sign weights can cancel each other out and zero weights do not increase the accumulator size.

This fact can be leveraged to train a network with more neurons, while not overflowing the accumulator, using a technique called [pruning](../explanations/pruning.md) where the developer can impose a number of zero-valued weights. Torch [provides support for pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) out of the box.

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

When pruning a larger neural network during training, it is easier to obtain a low bit-width accumulator while maintaining better final accuracy. Thus, pruning is more robust than training a similar, smaller network.
