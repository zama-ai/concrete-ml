# Utility functions for LoRA finetuning notebook

from torch.nn import Embedding
from transformers import Conv1D

from concrete.ml.torch.lora import CustomConv1D


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
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def replace_conv1d(module, module_index_to_skip=0):
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            # Skip the module if the index has not been reached, and decrement the index
            if module_index_to_skip >= 0:
                module_index_to_skip -= 1
            else:
                custom_linear = CustomConv1D(child.weight, bias=child.bias)
                setattr(module, name, custom_linear)
        else:
            module_index_to_skip = replace_conv1d(child, module_index_to_skip=module_index_to_skip)

    return module_index_to_skip


def get_remote_names(model, include_embedding_layers=False):
    remote_names = []
    for name, module in model.named_modules():
        # Exclude the backward module from the remote names
        # This is done on the client side
        if (
            isinstance(module, Conv1D)
            or include_embedding_layers
            and (isinstance(module, Embedding) or "lm_head" in name)
        ):
            remote_names.append(name)
        elif isinstance(module, CustomConv1D):
            remote_names.append(name + ".forward_module")
            remote_names.append(name + ".backward_module")
    return remote_names


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
