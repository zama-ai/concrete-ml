import json
import math
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from my_secrets import HF_TOKEN
from tqdm import tqdm
from transformers import AutoTokenizer

# Random Seed
SEED = 0

# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
NB_REMOTE_MODULES = 221
FREEZE_WEIGTHS = True
N_BITS = 7
MODE = f"{N_BITS}bit"

# Dataset Configuration
DATASET_NAME = "microsoft/orca-math-word-problems-200k"
MAX_LENGTH = 64
BATCH_SIZE = 4

# Logging / Outputs
TRAIN_LOG_FILE = f"training_log_{MODE}.txt"
EVAL_RESPONSES_FILE = f"eval_generated_responses_{MODE}.txt"

# Prompt Example
PROMPT = "When you multiply a number by 7, it becomes 98. What is that number?\n"

# Paths
DATA_DIR_PATH = Path("Data")
TRAIN_PATH = DATA_DIR_PATH / "train_dataset"
TEST_PATH = DATA_DIR_PATH / "test_dataset"

COMPILED_MODELS_PATH = Path("compiled_models/meta-llama/")
PATH_TO_CLIENTS = COMPILED_MODELS_PATH
PATH_TO_CLIENTS_KEYS = Path("compiled_models/meta-llama_keys")

# Devicepyth
DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")


# Set the tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# Ensure tokenizer has a pad token
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token
VOCAB_SIZE = TOKENIZER.vocab_size


def set_seed(seed):
    """Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def purge_compiled_model_dir(path_dir: Path, delete: bool = False) -> None:
    if delete and path_dir.exists():
        print(f"Clean `{path_dir}`")
        shutil.rmtree(path_dir)


def get_random_inputset(vocab_size, batch_size, max_length, device, dtype=torch.long):
    # Prepare input data for calibration
    inputs = torch.randint(0, vocab_size, (batch_size, max_length), dtype=dtype).to(device)
    labels = torch.randint(0, vocab_size, (batch_size, max_length), dtype=dtype).to(device)
    attention_mask = torch.ones((batch_size, max_length), dtype=dtype).to(device)

    inputset = {"input_ids": inputs, "attention_mask": attention_mask, "labels": labels}

    return inputset


def get_limited_batches(dataloader, num_batches=5):
    """Get a limited number of batches from dataloader."""
    limited_batches = []
    for i, batch in enumerate(dataloader):
        if i < num_batches:
            limited_batches.append(batch)
        else:
            break
    return limited_batches


def get_device(force_device="cpu"):
    if force_device == "cuda":
        if torch.cuda.is_available():
            # TODO add checks for concrete-python gpu
            return "cuda"
        # else Raise error
    return "cpu"


def causal_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def generate_and_print(prompt, model, tokenizer, seed=None, max_new_tokens=30):
    """
    Generates text based on the provided prompt and prints both the prompt and the generated text.

    Args:
        prompt (str): The input prompt to generate text from.
        model: The pre-trained language model.
        tokenizer: The tokenizer associated with the model.
        seed (int, optional): Seed for random number generators to ensure reproducibility.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 30.
    Returns:
        str: The generated text (response only, without the prompt).
    """
    try:
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

        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
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
        print(f"Prompt: `{prompt}`")
        print(f"Response: `{generated_text}`\n")

        return generated_text

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        return None


def metric_fn(model, dataloader, prompt, eval_responses_files, device):
    model.eval()
    model.to(device)
    total_loss, total_tokens, results = 0.0, 0, []
    response = generate_and_print(prompt, model, TOKENIZER, seed=SEED)
    if response:
        results.append({"prompt": prompt, "response": response})

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            valid = batch_labels[..., 1:] != -100
            loss = F.cross_entropy(
                outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
                batch_labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += valid.sum().item()
    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    with open(eval_responses_files, "a", encoding="utf-8") as f:
        f.write(f"Perplexity: {perplexity:.2f}\n")
        for i, r in enumerate(results):
            f.write(
                f"== Generation {i+1} ==\nPrompt:\n{r['prompt']}\n\nResponse:\n{r['response']}\n"
            )
            f.write("=" * 40 + "\n")

    return {"perplexity": perplexity}


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, examples):
        inputs, attention_masks, labels = [], [], []
        max_length = max(len(example["input_ids"]) for example in examples)
        for example in examples:
            inputs.append(example["input_ids"])
            attention_masks.append(example["attention_mask"])
            labels.append(example["labels"])

        def pad(sequences, value):
            return [x + [value] * (max_length - len(x)) for x in sequences]

        return {
            "input_ids": torch.tensor(pad(inputs, self.pad_id)),
            "attention_mask": torch.tensor(pad(attention_masks, 0)),
            "labels": torch.tensor(pad(labels, -100)),
        }


def quantize_remote_layers():
    weights_candidates = list(
        COMPILED_MODELS_PATH.glob("inference_model.*/*/server/remote_weights_layer*.npy")
    )
    weights_infos_candidates = list(
        COMPILED_MODELS_PATH.glob("inference_model.*/*/server/info*.json")
    )

    assert (
        len(weights_candidates) == NB_REMOTE_MODULES
    ), f"{len(weights_candidates)=} - {NB_REMOTE_MODULES=}"

    if len(weights_candidates) != len(weights_infos_candidates):
        raise ValueError("Le nombre de poids et d'info JSON ne correspond pas.")

    for path_weight, path_info in tqdm(zip(weights_candidates, weights_infos_candidates)):

        weights = np.load(path_weight).astype(np.float64)

        # Quantization
        weight_q, weight_scale, weight_zp, sum_w = per_channel_weight_quantization(
            weights.T, n_bits=7
        )
        with path_info.open("r") as f:
            info_data = json.load(f)

        info_data.update(
            {
                "weight_scale": weight_scale.cpu().numpy().tolist(),
                "weight_zp": weight_zp.cpu().numpy().tolist(),
                "sum_w": sum_w.cpu().numpy().tolist(),
            }
        )

        with path_info.open("w") as f:
            json.dump(info_data, f, indent=2)

        match = re.search(r"layer(\d+)", path_weight.name)
        layer_index = int(match.group(1))
        weights_quantized_path = (
            path_weight.parent / f"remote_quantized_weights_layer{layer_index}.npy"
        )
        np.save(weights_quantized_path, weight_q.cpu().numpy())
