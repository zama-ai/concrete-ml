# Encrypted fine-tuning

This section explains how to fine-tune neural-network models and large
language-models on private data. Small models can be be fine-tuned
using a single-client/single-server setup. For optimal
latency when fine-tuning larger models (e.g., GPT2 and bigger)
you should consider distributed computation, with multiple worker nodes performing the
training on encrypted data.

## Overview

{% hint style="info" %}
For a tutorial about applying FHE LORA fine-tuning to a small neural network, see [this notebook](../advanced_examples/LoraMLP.ipynb).
{% endhint %}

Concrete ML supports LORA, a parameter efficient fine-tuning (PEFT) approach, in
the [hybrid model](../guides/hybrid-models.md) paradigm. LORA adds
adapters, which contain a low number of fine-tunable weights, to the linear layers
in an original model.

Concrete ML will outsource the logic of a model's original forward and backward passes
to one or more remote servers. On the other hand, the forward and backward passes
over the LORA weights, the loss computation and the weight updates are performed
by the client side. As the number of LORA weights is low, this does not incur
significant added computation time for the model training client machine. More than
99% of a model's weights can be outsourced for large LLMs.

The main benefit of hybrid-model LORA training is outsourcing the computation of the
linear layers. In LLMs these layers have considerable size and performing inference
and gradient computations for them requires significant hardware. Using Concrete ML,
these computations can be securely outsourced, eliminating the memory bottleneck that
previously constrained such operations.

## Usage

Concrete ML integrates with the [`peft` package](https://huggingface.co/docs/peft/index)
which adds LORA layer adapters to a model's linear layers. Here are the steps to convert
a model to hybrid FHE LORA training.

### 1. Apply the `peft` LORA layers

The `LoraConfig` class from the `peft` package contains the various LORA parameters. It
is possible to specify which layers have LORA adapters through the `target_modules` argument.
Please refer to the
[`LoraConfig`](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig)
documentation for a reference on the various config options.

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

Concrete ML requires a conversion step for the `peft` model, adding
FHE compatible layers. In this step the number of gradient accumulation steps
can also be set. For LORA it is common to accumulate gradients over
several gradient descent steps before updating weights.

<!--pytest-codeblocks:cont-->

```python
lora_training = LoraTraining(peft_model)


# Update training parameters, including loss function
lora_training.update_training_parameters(
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=0.01),
    loss_fn=nn.CrossEntropyLoss(),
    training_args={"gradient_accumulation_steps": 1},
)

```

### 3. Compile a hybrid FHE model for the LORA adapted PyTorch model

Next, a hybrid FHE model must be compiled in order to convert
the selected outsourced layers to use FHE. Other layers
will be executed on the client side. The back-and-forth communication
of encrypted activations and gradients may require significant bandwidth.

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

Finally, the hybrid model can be trained, much in the same way
a PyTorch model is trained. The client is responsible for generating and iterating
on training data batches.

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

One fine-tuned, the LORA hybrid FHE model can perform inference only, through the
`model.inference_model` attribute of the hybrid FHE model.

<!--pytest-codeblocks:skip-->

```python
hybrid_model.model.inference_model(x)
```

### Toggle LORA layers

To compare to the original model, it is possible to disable the LORA weights
in order to use  the original model for inference.

<!--pytest-codeblocks:skip-->

```python
hybrid_model.model.inference_model.disable_adapter_layers()
hybrid_model.model.inference_model(x)

# Re-enable the LORA weights
hybrid_model.model.inference_model.enable_adapter_layers()
```
