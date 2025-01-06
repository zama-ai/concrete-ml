# Encrypted fine-tuning

This document explains how to fine-tune neural-network models and large language-models (LLMs) on private data.

Small models can be fine-tuned using a single-client/single-server setup. For larger models (such as GPT-2 and above), consider using distributed computation across multiple worker nodes to perform training on encrypted data for optimal latency.

## Overview

{% hint style="info" %}
Refer to [this notebook](../advanced_examples/LoraMLP.ipynb) to see the tutorial about applying FHE LoRA fine-tuning to a small neural network.
{% endhint %}

Concrete ML supports LoRA, a parameter-efficient fine-tuning (PEFT) approach, in the [hybrid model](../guides/hybrid-models.md) paradigm. LoRA adds adapter layers, which contain a small number of trainable parameters, to the linear layers of a base model.

In this setup, Concrete ML outsources the computationally intensive parts of forward and backward passes for large models to one or more remote servers. The training client machine only handles the LoRA-adapter forward/backward passes, loss computation, and adapter weight updates. Since the LoRA adapters are small, this additional computation on the client side is minimal. For large LLMs, over 99% of the model's weights can remain outsourced.

The main benefit of hybrid-model LoRA training is outsourcing the computation of linear layers, which are typically large in LLMs. These layers require substantial hardware for inference and gradient computation. By securely outsourcing this work, Concrete ML removes the memory bottleneck that previously limited such operations.

## Usage

Concrete ML integrates with the [`peft` package](https://huggingface.co/docs/peft/index) to add LoRA adapters to a model's linear layers. Below are the steps to convert a model into a hybrid FHE LoRA training setup.

### 1. Apply the `peft` LoRA layers

The `LoraConfig` class from the `peft` package contains the various LoRA parameters. You can specify which layers have LoRA adapters through the `target_modules` argument.
For a detailed reference of the various configuration options, refer to the
[`LoraConfig`](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig)
documentation.

```python
import torch
import torch.nn.functional as F
from torch import nn, optim
from peft import LoraConfig, get_peft_model
from concrete.ml.torch.lora import LoraTrainer
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

    def forward(self, x, labels=None):
        """Forward pass of the MLP."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create an initial model
model = SimpleMLP()

# Apply LoRA configuration
lora_config = LoraConfig(
    r=1,
    lora_alpha=1,
    lora_dropout=0.01,
    target_modules=["fc1", "fc2"],
    bias="none"
)

peft_model = get_peft_model(model, lora_config)

# Generate a second data-set for demonstration purposes
X_task2, y_task2 = make_circles(n_samples=32, noise=0.2, factor=0.5)
train_loader_task2 = DataLoader(
    TensorDataset(torch.Tensor(X_task2), torch.LongTensor(y_task2)),
    batch_size=32,
    shuffle=True
)
```

### 2. Convert the LoRA model to use custom Concrete ML layers

Next, we need to integrate the LoRA-adapted `peft_model` into the Concrete ML hybrid FHE training framework. This is done using the `LoraTrainer` class, which handles the logic of encrypting outsourced computations, running the forward and backward passes, and updating the LoRA adapter weights.

You can configure:

- The loss function.
- The optimizer and its parameters.
- Gradient accumulation steps (if needed).

<!--pytest-codeblocks:cont-->

```python
# Define a simple loss function
def simple_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)

# Create an Adam optimizer
optimizer = optim.Adam(peft_model.parameters(), lr=1e-3)

# Initialize trainer with the loss and optimizer
lora_trainer = LoraTrainer(
    peft_model,
    optimizer=optimizer,
    loss_fn=simple_loss,
)
```

### 3. Compile a hybrid FHE model for the LoRA adapted PyTorch model

Before training in FHE, we need to compile the model. Compilation calibrates and converts the outsourced linear layers to their FHE equivalents. The compile method uses representative data for this step.

<!--pytest-codeblocks:cont-->

```python
# Build a representative data-set for compilation
inputset = (
    torch.Tensor(X_task2[:16]),
    torch.LongTensor(y_task2[:16]),
)

# Calibrate and compile the model with 8-bit quantization
lora_trainer.compile(inputset, n_bits=8)
```

At this point, the trainer has a hybrid FHE model ready for encrypted execution of the outsourced layers. The LoRA layers remain on the client side in the clear.

### 4. Train the model on private data

You can now train the hybrid FHE model with your private data. The train method will run forward and backward passes, updating only the LoRA adapter weights locally while securely outsourcing the main layers’ computations.

<!--pytest-codeblocks:cont-->

```python
# Train in FHE mode
lora_trainer.train(train_loader_task2, fhe="execute")
```

## Additional options

### Inference

Once fine-tuned, the LoRA hybrid FHE model can perform inference only, through the
`peft_model` attribute of the hybrid FHE model.

<!--pytest-codeblocks:skip-->

```python
peft_model(x)
```

### Toggle LoRA layers

To compare to the original model, you can disable the LoRA weights to use the original model for inference.

<!--pytest-codeblocks:skip-->

```python
peft_model.disable_adapter_layers()
peft_model(x)

# Re-enable the LoRA weights
peft_model.enable_adapter_layers()
```
