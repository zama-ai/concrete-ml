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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from utils_lora import generate_and_print

from concrete.ml.torch.lora import LoraTrainer

# Set seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Freeze the original model weights
for param in model.parameters():
    param.requires_grad = False

# Print initial generation with base model
PROMPT = "What is 2+2?\n"
print("Initial generation with base model:")
print(generate_and_print(PROMPT, model, tokenizer, seed=SEED))

# Apply LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
peft_model = get_peft_model(model, peft_config)

# Load the dataset
raw_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")

MAX_LENGTH = 128


def processed(example):
    # Combine question and answer into a single prompt
    qa_text = example["question"].strip() + "\n" + example["answer"].strip()
    tokens = tokenizer(
        qa_text, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors=None
    )

    # Discard examples longer than MAX_LENGTH (though truncation is applied)
    if len(tokens["input_ids"]) > MAX_LENGTH:
        return {}

    # Determine question length
    question_tokens = tokenizer(
        example["question"].strip(), truncation=True, max_length=MAX_LENGTH, padding=False
    )
    question_length = len(question_tokens["input_ids"])

    # Add the newline length
    newline_tokens = tokenizer("\n", add_special_tokens=False)["input_ids"]
    question_boundary = question_length + len(newline_tokens)

    # Create labels and mask the question part
    labels = tokens["input_ids"].copy()
    question_boundary = min(question_boundary, len(labels))  # Safety check
    for i in range(question_boundary):
        labels[i] = -100

    tokens["labels"] = labels
    return tokens


# Apply preprocessing
tokenized_dataset = raw_dataset.map(
    processed, batched=False, remove_columns=raw_dataset.column_names
)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

# Split into train/test
split_dataset = tokenized_dataset.train_test_split(test_size=0.33, seed=SEED)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 4
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
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
    logging_steps=50,
    # evaluation_strategy is handled by LoraTrainer, not HuggingFace Trainer
    report_to="none",
)


# Define a causal LM loss function
def causal_lm_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction="mean"
    )
    return loss


# Metric function for evaluation
def metric_fn(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    print("\nModel response during evaluation:")
    print(generate_and_print(PROMPT, model, tokenizer, seed=SEED))

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to("cpu")
            labels = batch["labels"].to("cpu")
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to("cpu")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Compute loss as in causal LM
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_positions = shift_labels != -100

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += valid_positions.sum().item()

            current_perplexity = (
                math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
            )
            progress_bar.set_postfix({"perplexity": f"{current_perplexity:.2f}"})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    return {"perplexity": perplexity}


# Create a HuggingFace Trainer instance to get optimizer and scheduler
hf_trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
train_dataloader = hf_trainer.get_train_dataloader()
hf_trainer.create_optimizer_and_scheduler(num_training_steps=len(train_dataloader) * EPOCHS)

optimizer = hf_trainer.optimizer
lr_scheduler = hf_trainer.lr_scheduler

# Prepare input data for calibration
BLOCK_SIZE = MAX_LENGTH
input_tensor = torch.randint(
    0, tokenizer.vocab_size, (PER_DEVICE_TRAIN_BATCH_SIZE, BLOCK_SIZE), dtype=torch.long
)
label_tensor = torch.randint(
    0, tokenizer.vocab_size, (PER_DEVICE_TRAIN_BATCH_SIZE, BLOCK_SIZE), dtype=torch.long
)
attention_mask = torch.ones((PER_DEVICE_TRAIN_BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
inputset = (input_tensor, label_tensor, attention_mask)

# Prepare eval loader
eval_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)

lora_trainer = LoraTrainer(
    model=peft_model,
    optimizer=optimizer,
    loss_fn=causal_lm_loss,
    lr_scheduler=lr_scheduler,
    training_args=vars(training_args),
    n_layers_to_skip_for_backprop=3,
    eval_loader=eval_loader,
    eval_metric_fn=metric_fn,
    logging_steps=1,
    eval_steps=10,
    train_log_path="training_log.txt",
)

# Compile the model with FHE
lora_trainer.compile(inputset, n_bits=16)

# Train the model using LoraTrainer
print("Starting training using LoraTrainer...")
lora_trainer.train(train_dataloader, num_epochs=EPOCHS, fhe="disable")

# After training, retrieve all training losses
all_losses = lora_trainer.get_training_losses()
print("Recorded training losses:", all_losses)

# Evaluate the original model (disabling adapter layers)
peft_model.disable_adapter_layers()
orig_metrics = metric_fn(peft_model, eval_loader)
print("Evaluation on original layers (adapter disabled):", orig_metrics)

# Evaluate the fine-tuned model (enabling adapter layers)
peft_model.enable_adapter_layers()
finetuned_metrics = metric_fn(peft_model, eval_loader)
print("Evaluation on fine-tuned model (adapter enabled):", finetuned_metrics)

# Compare generation before and after fine-tuning
peft_model.disable_adapter_layers()
print("Original model generation:")
print(generate_and_print(PROMPT, peft_model, tokenizer, seed=SEED))

peft_model.enable_adapter_layers()
print("Fine-tuned model generation:")
print(generate_and_print(PROMPT, peft_model, tokenizer, seed=SEED))

# Save the fine-tuned model
save_path = Path("deployment/gpt2_lora_finetuned")
if save_path.is_dir() and any(save_path.iterdir()):
    shutil.rmtree(save_path)
lora_trainer.save_and_clear_private_info(save_path)
print("Model saved to:", save_path)
