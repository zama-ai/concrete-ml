# Encrypted fine-tuning

This document explains how to fine-tune neural-network models and large language-models(LLMs) on private data.

Small models can be fine-tuned using a single-client/single-server setup. For larger models (such as GPT-2 and above), consider using distributed computation across multiple worker nodes to perform training on encrypted data for optimal latency.

## Overview

{% hint style="info" %}
Refer to [this notebook](../advanced_examples/LoraMLP.ipynb) to see the tutorial about applying FHE LORA fine-tuning to a small neural network.
{% endhint %}

Concrete ML supports LORA, a parameter efficient fine-tuning (PEFT) approach, in the [hybrid model](../guides/hybrid-models.md) paradigm. LORA adds adapters, which contain a low number of fine-tunable weights, to the linear layers in an original model.

In this setup, Concrete ML outsources the forward and backward passes of the model's original logic to one or more remote servers. Meanwhile, the forward and backward passes over the LORA weights, the loss computation and the weight updates are performed by the client side. As the number of LORA weights is low, this does not significantly increase the computational load for the model training client machine. For large LLMs, over 99% of the model's weights can be outsourced.

The main benefit of hybrid-model LORA training is outsourcing the computation of linear layers, which are typically large in LLMs. These layers require substantial hardware for inference and gradient computation. By securely outsourcing this work, Concrete ML removes the memory bottleneck that previously limited such operations.

## Usage

Concrete ML integrates with the [`peft` package](https://huggingface.co/docs/peft/index),
which adds LORA layer adapters to a model's linear layers. Here are the steps to convert
a model to hybrid FHE LORA training.

### 1. Apply the `peft` LORA layers

The `LoraConfig` class from the `peft` package contains the various LORA parameters. You can specify which layers have LORA adapters through the `target_modules` argument.
For a detailed reference of the various configuration options, refer to the
[`LoraConfig`](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig)
documentation.

```python
import torch
from torch import nn, optim
from peft import LoraConfig, get_peft_model
from concrete.ml.torch.lora import LoraTraining, get_remote_names
from concrete.ml.torch.hybrid_model import HybridFHEModel
from sklearn.datasets import make_circles
from torch.utils.data import DataLoader, TensorDataset

class SimpleMLP(nn.Module):
    """Simple MLP model without LoRA layers."""

    def __init__(self, input_size=2, hidden_size=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Forward pass of the MLP."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

lora_config = LoraConfig(
    r=1, lora_alpha=1, lora_dropout=0.01, target_modules=["fc1", "fc2"], bias="none"
)

model = SimpleMLP()
# The initial training loop of the model should be
# added at this point on an initial data-set

# A second data-set, task2 is generated
X_task2, y_task2 = make_circles(n_samples=32, noise=0.2, factor=0.5)
train_loader_task2 = DataLoader(
    TensorDataset(torch.Tensor(X_task2), torch.LongTensor(y_task2)),
    batch_size=32,
    shuffle=True
)

# Apply LoRA to the model
peft_model = get_peft_model(model, lora_config)
```

### 2. Convert the LORA model to use custom Concrete ML layers

Concrete ML requires converting the `peft` model to add
FHE compatible layers. In this step, you can configure several fine-tuning
parameters:

- The number of gradient accumulation steps: LORA commonly accumulate gradients over several gradient descent steps before updating weights.
- The optimizer parameters
- The loss function

<!--pytest-codeblocks:cont-->

```python
lora_training = LoraTraining(peft_model)
```

### 3. Compile a hybrid FHE model for the LORA adapted PyTorch model

Compile the hybrid FHE model to convert the selected outsourced layers to use FHE, while the rest will run on the client side. Note that the exchange of encrypted activations and gradients may require significant bandwidth.

<!--pytest-codeblocks:cont-->

```python
# Find layers that can be outsourced
remote_names = get_remote_names(lora_training)

# Build the hybrid FHE model
hybrid_model = HybridFHEModel(lora_training, module_names=remote_names)

# Build a representative data-set for compilation
inputset = (
    torch.Tensor(X_task2[:16]),
    torch.LongTensor(y_task2[:16]),
)

# Calibrate and compile the model
hybrid_model.model.toggle_calibrate(enable=True)
hybrid_model.compile_model(inputset, n_bits=8)
hybrid_model.model.toggle_calibrate(enable=False)
```

### 4. Train the model on private data

Finally, the hybrid model can be trained, similar to training a PyTorch model. The client handles training data batches generation and iteration.

<!--pytest-codeblocks:cont-->

```python
# Assume train_loader is a torch.DataLoader

hybrid_model.model.inference_model.train()
hybrid_model.model.toggle_run_optimizer(enable=True)

for x_batch, y_batch in train_loader_task2:
    loss, _ = hybrid_model((x_batch, y_batch), fhe="execute")
```

## Additional options

### Inference

Once fine-tuned, the LORA hybrid FHE model can perform inference only, through the
`model.inference_model` attribute of the hybrid FHE model.

<!--pytest-codeblocks:skip-->

```python
hybrid_model.model.inference_model(x)
```

### Toggle LORA layers

To compare to the original model, you can disable the LORA weights to use the original model for inference.

<!--pytest-codeblocks:skip-->

```python
hybrid_model.model.inference_model.disable_adapter_layers()
hybrid_model.model.inference_model(x)

# Re-enable the LORA weights
hybrid_model.model.inference_model.enable_adapter_layers()
```
