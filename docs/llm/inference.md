# Encrypted LLM Inference

LLMs can be converted to use FHE to generate encrypted tokens based on encrypted prompts. Concrete ML implements LLM inference as a client/server protocol where

- The client executes non-linear layers in the LLM, such as attention and activation functions.
- The server executes linear layers, such as projection and embedding.

The FHE LLM implementation in Concrete ML has the following characteristics:

- Data transfer is necessary for each linear layer. The size of encrypted data
  is about 4x the size of the clear data that are input/outputs to the linear layers. For instance:
  - A [LLAMA 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model exchanges around 18MB of data per token.
  - A [GPT2](https://huggingface.co/openai-community/gpt2) mode exchanges around 2.2MB of data per token.
- The client machine needs to perform some computation, thus it needs to execute some PyTorch layers.
- Advantages of FHE include:
  - Offloading computation from clients with limited hardware.
  - Preserving intellectual property by running sensitive model components on encrypted data.

## Compiling an LLM for FHE Inference

This document introduces how to use Concrete ML to run encrypted LLM inference with FHE.
To prepare an LLM model for FHE inference, use the `HybridFHEModel` class:

```python
import random
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D, Trainer, TrainingArguments

from concrete.ml.torch.hybrid_model import HybridFHEModel

# Load the GPT2 model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

# Determine which layers run with FHE (all linear ones)
remote_names = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, Conv1D)):
        remote_names.append(name)

# Create the HybridFHEModel with the specified remote modules
hybrid_model = HybridFHEModel(model, module_names=remote_names)

# Prepare input data for calibration
input_tensor = torch.randint(0, tokenizer.vocab_size, (1, 32), dtype=torch.long)

# Calibrate and compile the model
hybrid_model.compile_model(input_tensor, n_bits=8, use_dynamic_quantization=True)
```

After `compile_model` is called as above, you can retrieve the FHE-enabled model in
`hybrid_model.model`.

As for all Concrete ML models, to verify accuracy of the converted LLM on clear data, you can use `fhe='disable'` or `fhe='simulate'`. To actually executed on
encrypted data, set the `fhe_mode` to `execute`:

<!--pytest-codeblocks:cont-->

```python
hybrid_model.set_fhe_mode("execute")
```

Next, to generate some tokens using FHE computation, run:

<!--pytest-codeblocks:cont-->

```python
prompt = "Programming is"
inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}

N_TOKENS_GENERATE = 1
# Generate text
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=N_TOKENS_GENERATE,
        top_p=0.9,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# Get only the newly generated tokens
input_length = inputs["input_ids"].shape[1]
generated_ids = output[0, input_length:]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# Print the prompt and generated text
print(f"Prompt: {prompt}")
print(f"Response: {generated_text}\n")
```

## Latency and throughput

The Concrete ML LLM model inference, as described above, can use GPUs to obtain acceleration. Running on GPU reduces latency by ~30x. For example, generating
a GPT2 token on GPU takes ~11 seconds, while it takes ~300 seconds.
