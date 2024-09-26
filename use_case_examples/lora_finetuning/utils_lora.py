# Utility functions for LoRA finetuning notebook

from torch.nn import Embedding
from transformers import Conv1D


def generate_text(prompt, model, tokenizer, max_new_tokens=30):
    # Encode the input prompt
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
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
    return generated_text


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
