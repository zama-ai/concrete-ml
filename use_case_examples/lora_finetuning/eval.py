import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from utils_lora import generate_and_print

from concrete.ml.torch.lora import LoraTrainer


# ------------------------------------------------------------------
# Command-Line Args (Select Mode)
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["torch", "8bit", "16bit"], default="torch")
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Number of steps between evaluations"
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force CPU execution even if GPU is available"
    )
    return parser.parse_args()


args = parse_args()
mode_str = args.mode
EVAL_RESPONSES_FILE = f"eval_generated_responses_{mode_str}.txt"
TRAIN_LOG_FILE = f"training_log_{mode_str}.txt"
SAVE_PATH = Path(f"deployment/llama_lora_finetuned_{mode_str}")


# ------------------------------------------------------------------
# Basic Setup: Device, Seed, Model/Tokenizer
# ------------------------------------------------------------------
def get_device():
    if args.force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
for p in model.parameters():
    p.requires_grad = False

# Test base model output
PROMPT = "When you multiply a number by 7, it becomes 98. What is that number?\n"
generate_and_print(PROMPT, model, tokenizer, seed=SEED)

# ------------------------------------------------------------------
# LoRA Configuration
# ------------------------------------------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
peft_model = get_peft_model(model, peft_config)

# ------------------------------------------------------------------
# Dataset & Preprocessing
# ------------------------------------------------------------------
MAX_LENGTH = 128
raw_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")


# First, filter out samples that exceed length
def length_filter(example):
    q_len = len(tokenizer(example["question"], add_special_tokens=False)["input_ids"])
    a_len = len(tokenizer(example["answer"], add_special_tokens=False)["input_ids"])
    return (q_len + a_len + 1) <= MAX_LENGTH


filtered_dataset = raw_dataset.filter(length_filter)


# Print length distribution statistics
def get_lengths(example):
    q_len = len(tokenizer(example["question"], add_special_tokens=False)["input_ids"])
    a_len = len(tokenizer(example["answer"], add_special_tokens=False)["input_ids"])
    total_len = q_len + a_len + 1  # +1 for newline
    return {"q_len": q_len, "a_len": a_len, "total_len": total_len}


lengths = filtered_dataset.map(get_lengths)
q_lengths = [x["q_len"] for x in lengths]
a_lengths = [x["a_len"] for x in lengths]
total_lengths = [x["total_len"] for x in lengths]

print("\nLength Distribution Statistics:")
print(f"Original dataset size: {len(raw_dataset):,}")
print(f"Filtered dataset size: {len(filtered_dataset):,}")
print(f"Percentage kept: {100 * len(filtered_dataset)/len(raw_dataset):.1f}%\n")
print("Question lengths: ")
print(f"  Min: {min(q_lengths)}, Max: {max(q_lengths)}")
print(f"  Mean: {sum(q_lengths)/len(q_lengths):.1f}")
print(f"  Median: {sorted(q_lengths)[len(q_lengths)//2]}")
print("\nAnswer lengths:")
print(f"  Min: {min(a_lengths)}, Max: {max(a_lengths)}")
print(f"  Mean: {sum(a_lengths)/len(a_lengths):.1f}")
print(f"  Median: {sorted(a_lengths)[len(a_lengths)//2]}")
print("\nTotal lengths (including newline):")
print(f"  Min: {min(total_lengths)}, Max: {max(total_lengths)}")
print(f"  Mean: {sum(total_lengths)/len(total_lengths):.1f}")
print(f"  Median: {sorted(total_lengths)[len(total_lengths)//2]}\n")


def process_example(example):
    """Tokenize & create labels that hide the question portion (i.e., label=-100)."""
    question = example["question"].strip()
    answer = example["answer"].strip()

    # Tokenize full Q + A (joined by newline)
    tokens = tokenizer(
        question + "\n" + answer,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Identify how many tokens belong to the question, plus 1 for the newline
    question_length = len(tokenizer(question, add_special_tokens=False)["input_ids"]) + 1

    # Create labels
    labels = tokens["input_ids"].copy()
    # Mask out the question (so we don't compute loss there)
    for i in range(question_length):
        if i < len(labels):
            labels[i] = -100
    tokens["labels"] = labels

    return tokens


# Map with our preprocessing
tokenized_dataset = filtered_dataset.map(
    process_example,
    batched=False,
    remove_columns=filtered_dataset.column_names,
)

N_TRAIN_SAMPLES = 10000
tokenized = tokenized_dataset.train_test_split(test_size=1000, seed=SEED, shuffle=True)
train_dataset, test_dataset = tokenized["train"], tokenized["test"]
train_dataset = train_dataset.select(range(min(N_TRAIN_SAMPLES, len(train_dataset))))
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


# ------------------------------------------------------------------
# Collator (Pads & Preserves -100)
# ------------------------------------------------------------------
class DataCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, examples):
        inputs, attention_masks, labels = [], [], []
        max_length = 0
        for example in examples:
            inputs.append(example["input_ids"])
            attention_masks.append(example["attention_mask"])
            labels.append(example["labels"])
            max_length = max(max_length, len(example["input_ids"]))

        def pad(sequences, value):
            return [x + [value] * (max_length - len(x)) for x in sequences]

        return {
            "input_ids": torch.tensor(pad(inputs, self.pad_id)),
            "attention_mask": torch.tensor(pad(attention_masks, 0)),
            "labels": torch.tensor(pad(labels, -100)),
        }


collator = DataCollator(tokenizer)

# ------------------------------------------------------------------
# Training Arguments
# ------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    save_total_limit=1,
    use_cpu=True,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    seed=SEED,
    data_seed=SEED,
    warmup_steps=10,
    weight_decay=0.01,
    prediction_loss_only=True,
    report_to="none",
)


# ------------------------------------------------------------------
# Loss Function & Metric
# ------------------------------------------------------------------
def causal_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def metric_fn(model, dataloader):
    model.eval()
    total_loss, total_tokens, results = 0.0, 0, []
    response = generate_and_print(PROMPT, model, tokenizer, seed=SEED)
    if response:
        results.append({"prompt": PROMPT, "response": response})

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(DEVICE)
            batch_labels = batch["labels"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            # Check device of model and input_ids
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Input IDs device: {input_ids.device}")
            print(f"Batch labels device: {batch_labels.device}")
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

    with open(EVAL_RESPONSES_FILE, "a") as f:
        f.write(f"Perplexity: {perplexity:.2f}\n")
        for i, r in enumerate(results):
            f.write(
                f"== Generation {i+1} ==\nPrompt:\n{r['prompt']}\n\nResponse:\n{r['response']}\n"
            )
            f.write("=" * 40 + "\n")

    return {"perplexity": perplexity}


# ------------------------------------------------------------------
# Create Hugging Face Trainer to Setup Optimizer/Scheduler
# ------------------------------------------------------------------
hf_trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)
train_dl = hf_trainer.get_train_dataloader()
hf_trainer.create_optimizer_and_scheduler(len(train_dl) * training_args.num_train_epochs)
optimizer, lr_scheduler = hf_trainer.optimizer, hf_trainer.lr_scheduler

# ------------------------------------------------------------------
# Calibration (Dummy) Data
# ------------------------------------------------------------------
inputset = {
    "input_ids": torch.randint(0, tokenizer.vocab_size, (4, MAX_LENGTH)),
    "attention_mask": torch.ones((4, MAX_LENGTH), dtype=torch.long),
    "labels": torch.randint(0, tokenizer.vocab_size, (4, MAX_LENGTH)),
}

# ------------------------------------------------------------------
# Evaluation DataLoader
# ------------------------------------------------------------------
eval_dl = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collator)

# ------------------------------------------------------------------
# LoraTrainer Initialization
# ------------------------------------------------------------------
lora_trainer = LoraTrainer(
    model=peft_model,
    optimizer=optimizer,
    loss_fn=causal_lm_loss,
    lr_scheduler=lr_scheduler,
    training_args=vars(training_args),
    n_layers_to_skip_for_backprop=3,
    eval_loader=eval_dl,
    eval_metric_fn=metric_fn,
    logging_steps=1,
    eval_steps=args.eval_steps,
    train_log_path=TRAIN_LOG_FILE,
)

# ------------------------------------------------------------------
# Compile if 8bit/16bit
# ------------------------------------------------------------------
if args.mode == "torch":
    print("Using Torch mode (no compilation).")
else:
    bits = 8 if args.mode == "8bit" else 16
    print(f"Compiling model with {bits} bits...")
    lora_trainer.compile(inputset, n_bits=bits)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
open(EVAL_RESPONSES_FILE, "w").write("=== Training Start ===\n")
fhe_mode = "torch" if args.mode == "torch" else "disable"

# Set device of model to DEVICE
peft_model.to(DEVICE)

# Full evaluation before training
print("Evaluating quantized model before training...")
peft_model.disable_adapter_layers()
original_metrics = metric_fn(peft_model, eval_dl)
print("Original model metrics:", original_metrics)
peft_model.enable_adapter_layers()

print(f"\nTraining in {args.mode} mode...")
lora_trainer.train(train_dl, fhe=fhe_mode, device=DEVICE)
print("Training losses:", lora_trainer.get_training_losses())

# Full evaluation after training
print("\nEvaluating fine-tuned model...")
final_metrics = metric_fn(peft_model, eval_dl)
print("Fine-tuned model metrics:", final_metrics)

print("\nMetrics comparison:")
print(f"Original perplexity: {original_metrics['perplexity']:.2f}")
print(f"Fine-tuned perplexity: {final_metrics['perplexity']:.2f}")

# ------------------------------------------------------------------
# Compare Generations
# ------------------------------------------------------------------
peft_model.disable_adapter_layers()
print("Original model says:", generate_and_print(PROMPT, peft_model, tokenizer, seed=SEED))
peft_model.enable_adapter_layers()
print("Fine-tuned model says:", generate_and_print(PROMPT, peft_model, tokenizer, seed=SEED))

# ------------------------------------------------------------------
# Save Model
# ------------------------------------------------------------------
if SAVE_PATH.is_dir() and any(SAVE_PATH.iterdir()):
    shutil.rmtree(SAVE_PATH)
lora_trainer.save_and_clear_private_info(SAVE_PATH)
print("Model saved to:", SAVE_PATH)
