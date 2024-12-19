# Utility functions for LoRA finetuning notebook

import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers.generation.stopping_criteria import (  # Add this line
    StoppingCriteria,
    StoppingCriteriaList,
)


class NewlineStopping(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Get all token IDs that represent newline characters
        self.newline_tokens = set(
            [
                self.tokenizer.encode("\n")[0],
                self.tokenizer.encode("\r")[0] if len(self.tokenizer.encode("\r")) > 0 else None,
                (
                    self.tokenizer.encode("\r\n")[0]
                    if len(self.tokenizer.encode("\r\n")) > 0
                    else None
                ),
            ]
        )
        self.newline_tokens.discard(None)

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is a newline
        last_token = input_ids[0][-1].item()
        return last_token in self.newline_tokens


def generate_and_print(prompt, model, tokenizer, seed=None, max_new_tokens=30):
    """
    Generates text based on the provided prompt and prints both the prompt and the generated text.

    Args:
        prompt (str): The input prompt to generate text from.
        model: The pre-trained language model.
        tokenizer: The tokenizer associated with the model.
        seed (int, optional): Seed for random number generators to ensure reproducibility.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 30.
    """
    # Set the environment variable for CuBLAS deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Encode the input prompt
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt from the generated text if it is included
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :].strip()

    # Only keep text up to the first newline
    generated_text = generated_text.split("\n")[0]

    # Print the prompt and generated text on the same line
    print(f"{prompt} {generated_text}")


def print_weights_and_size(model, print_detail=False):
    total_weights = 0
    total_lora_weights = 0
    for name, param in model.named_parameters():
        total_weights += param.numel()

        if "lora" in name:
            total_lora_weights += param.numel()

        if print_detail:
            print(name, param.numel())

    print(f"Total number of weights: {total_weights}")
    print(f"Total number of LoRA weights: {total_lora_weights}")

    return total_weights
