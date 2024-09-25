# Encrypted fine-tuning

This section explains how to fine-tune neural-network models and large
language-models on private data. Small models can be be fine-tuned
using a single-client/single-server setup. For optimal
latency when fine-tuning larger models (e.g. GPT2 and bigger)
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
these computations can be securely outsourced.

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

<!--pytest-codeblocks:skip-->

```python
# Apply LoRA to the model
peft_model = get_peft_model(model, peft_config)

lora_config = LoraConfig(
    r=1, lora_alpha=1, lora_dropout=0.01, target_modules=["fc1", "fc2"], bias="none"
)
```

### 2. Convert the LORA model to use custom Concrete ML layers

Concrete ML requires a conversion step for the `peft` model, adding
FHE compatible layers. In this step the number of gradient accumulation steps
can also be set. For LORA it is common to accumulate gradients over
several gradient descent steps before updating weights.

<!--pytest-codeblocks:skip-->

`python`
gradient_accum_steps = 2
lora_training = LoraTraining(peft_model, gradient_accum_steps)

````

### 3. Compile a hybrid FHE model for the LORA adapted PyTorch model

Next, a hybrid FHE model must be compiled in order to convert 
the selected outsourced layers to use FHE. Other layers
will be executed on the client side. The back-and-forth communication
of encrypted activations and gradients may require significant bandwidth.

<!--pytest-codeblocks:skip-->
```python```
# Find layers that can be outsourced
remote_names = get_remote_names(lora_training)

# Build the hybrid FHE model
hybrid_model = HybridFHEModel(lora_training, module_names=remote_names)

# generate an inputset
# inputset = ...

# Calibrate and compile the model
hybrid_model.model.toggle_calibrate(enable=True)
hybrid_model.compile_model(inputset, n_bits=8)
hybrid_model.model.toggle_calibrate(enable=False)
````

### 4. Train the model on private data

Finally, the hybrid model can be trained, much in the same way
a PyTorch model is trained. The client is responsible for generating and iterating
on training data batches.

<!--pytest-codeblocks:skip-->

```python
# Assume train_loader is a torch.DataLoader

hybrid_model.model.inference_model.train()
hybrid_model.model.toggle_run_optimizer(enable=True)

for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    loss, _ = hybrid_model((x_batch, y_batch), fhe=fhe)
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
