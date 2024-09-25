# Encrypted fine-tuning

This section explains how to fine-tune neural-network models and large 
language-models on private data. Small models can be be fine-tuned 
using a single client/single server setup, but larger models (e.g. GPT2 and bigger)
require distributed computation, with multiple worker nodes performing the
training on encrypted data. 

## Overview

Concrete ML supports LORA, a parameter efficient fine-tuning (PEFT) approach, in 
the [hybrid model](../guides/hybrid-models.md) paradigm. LORA adds
adapters, which contain a low number of fine-tunable weights, to the linear layers 
in an original model. 

Concrete ML will outsource the computations in a model's original forward and backward passes
to one or more remote servers. On the other hand, the forward and backward passes
over the LORA weights, the loss computation and the weight updates are performed
by the client side. As the number of LORA weights is low, this does not incur
significant added computation time for the model training client machine. 

The main benefit of hybrid-model LORA training is outsourcing the computation of the 
linear layers. In LLMs these layers have considerable size and performing inference
and gradient computations for them requires significant hardware. Using Concrete ML,
these computations can be securely outsourced.

## Usage

Concrete ML integrates with the `peft` package which adds LORA layer adapters
to a model's linear layers. Here are the steps to follow to convert
a model to use hybrid FHE LORA training:

### 1. Apply the `peft` LORA layers

The `LoraConfig` class from the `peft` package contains the various LORA parameters. It 
is possible to specify which layers have LORA adapters through the `target_modules` argument.
Please refer to the [`LoraConfig`](https://huggingface.co/docs/peft/index) documentation
for a reference on the various config options.

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
```python```
gradient_accum_steps = 2
lora_training = LoraTraining(peft_model, gradient_accum_steps)
```

### 3. Compile a hybrid FHE model for the LORA adapted PyTorch model


- cml classes
- HF classes


