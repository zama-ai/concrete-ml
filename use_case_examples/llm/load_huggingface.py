from pathlib import Path
from typing import Union

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_hf_pretrained(
    dir_path: Union[Path, str], hf_object: type, **kwargs
) -> Union[GPT2LMHeadModel, GPT2Tokenizer]:
    """Get a pre-trained Hugging Face GPT-2 model or tokenizer.

    Args:
        dir_path (Union[Path, str]): The directory path where to download the pre-trained files.
        hf_object (type): The Hugging Face class to consider.
    """
    dir_path = Path(__file__).parent / dir_path

    # If the dir_path directory does not exists or is empty, download the files and save them
    if not dir_path.is_dir() or not any(dir_path.iterdir()):
        hf_gpt2 = hf_object.from_pretrained("gpt2", **kwargs)
        hf_gpt2.save_pretrained(dir_path)

    # Else, load the Hugging Face object
    else:
        hf_gpt2 = hf_object.from_pretrained(dir_path, **kwargs)

    return hf_gpt2


def get_gpt2_tokenizer(dir_path: Union[Path, str], **kwargs) -> GPT2Tokenizer:
    """Get a pre-trained Hugging Face GPT-2 tokenizer.

    Args:
        dir_path (Union[Path, str]): The directory path where to download the pre-trained files.

    Returns:
        GPT2Tokenizer: Hugging Face's GPT-2 tokenizer.
    """
    return get_hf_pretrained(dir_path, GPT2Tokenizer, **kwargs)


def get_gpt2_model(dir_path: Union[Path, str], **kwargs) -> GPT2LMHeadModel:
    """Get a pre-trained Hugging Face GPT-2 model.

    Uses 12 layers, 12 heads, 768 embedding dims, 50257 vocab size, 1024 max sequence length.

    model.config = {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
        ...
    }

    model.state_dict = {
        "transformer.wte.weight": tensor, (50257, 768)
        "transformer.wpe.weight": tensor, (1024, 768)

        For layer 0 to 11:
            "transformer.h.0.ln_1.weight": tensor, (768,)
            "transformer.h.0.ln_1.bias": tensor, (768,)

            "transformer.h.0.attn.bias": tensor, (1, 1, 1024, 1024)  # not used
            "transformer.h.0.ln_1.masked_bias": tensor, (,)  # not used

            "transformer.h.0.attn.c_attn.weight": tensor, (768, 2304)
            "transformer.h.0.attn.c_attn.bias": tensor, (2304,)
            "transformer.h.0.attn.c_proj.weight": tensor, (768, 768)
            "transformer.h.0.attn.c_proj.bias": tensor, (768,)

            "transformer.h.0.ln_2.weight": tensor, (768,)
            "transformer.h.0.ln_2.bias": tensor, (768,)

            "transformer.h.0.mlp.c_fc.weight": tensor, (768, 3072)
            "transformer.h.0.mlp.c_fc.bias": tensor, (3072,)

            "transformer.h.0.mlp.c_proj.weight": tensor, (3072, 768)
            "transformer.h.0.mlp.c_proj.bias": tensor, (768,)

            ...

        "transformer.ln_f.weight": tensor, (768,)
        "transformer.ln_f.bias": tensor, (768,)

        "lm_head.weight": tensor, (50257, 768)  # not used, same as "transformer.h.0.ln_1.weight"
    }

    Args:
        dir_path (Union[Path, str]): The directory path where to download the pre-trained files.

    Returns:
        GPT2LMHeadModel: Hugging Face's GPT2LMHeadModel.
    """
    return get_hf_pretrained(dir_path, GPT2LMHeadModel, **kwargs)
