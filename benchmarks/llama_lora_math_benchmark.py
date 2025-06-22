#!/usr/bin/env python3
"""LLama LoRA Fine-tuning Benchmark Script for Math Word Problems

This script runs LLama LoRA fine-tuning benchmarks on the Orca Math Word Problems dataset
and outputs metrics in the format expected by the benchmark database.
"""

import argparse
import datetime
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import cpuinfo
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset  # pylint: disable=import-error
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Add the use_case_examples directory to path for utils_lora
sys.path.append(str(Path(__file__).parent.parent / "use_case_examples" / "lora_finetuning"))
try:
    from utils_lora import generate_and_print
except ImportError:
    # Fallback implementation if utils_lora is not available
    def generate_and_print(prompt, model, tokenizer, seed=0, max_new_tokens=50):
        """Simple generation function if utils_lora is not available"""
        set_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        return response


# Try to import convert utilities
try:
    from convert import get_git_hash, get_git_hash_date, git_iso_to_python_iso, is_git_diff

    CONVERT_AVAILABLE = True
except ImportError:
    CONVERT_AVAILABLE = False


def get_size(bytes_count: float, suffix="B"):
    """Scale bytes to its proper format"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_count < factor:
            return f"{bytes_count:.2f} {unit}{suffix}"
        bytes_count /= factor
    return f"{bytes_count:.2f} {suffix}"


def get_system_information():
    """Get system information"""
    info = {}
    info["ram"] = get_size(psutil.virtual_memory().total)
    info["cpu"] = cpuinfo.get_cpu_info()["brand_raw"]
    info["os"] = f"{platform.system()} {platform.release()}"

    # Added metadata about the system
    info["platform"] = platform.system()
    info["platform-release"] = platform.release()
    info["platform-version"] = platform.version()
    info["architecture"] = platform.machine()
    info["hostname"] = socket.gethostname()
    info["processor"] = platform.processor()
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["total_cores"] = psutil.cpu_count(logical=True)
    uname = platform.uname()
    info["machine"] = uname.machine
    info["processor"] = uname.processor
    info["system"] = uname.system
    info["node_name"] = uname.node
    info["release"] = uname.release
    info["version"] = uname.version
    info["swap"] = get_size(psutil.swap_memory().total)

    # Check for GPU information
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = get_size(torch.cuda.get_device_properties(0).total_memory)

    return info


def get_ec2_metadata():
    """Get EC2 metadata if available"""
    res = {}
    try:
        output = subprocess.check_output("ec2metadata", shell=True, encoding="utf-8")
        for line in output.split("\n"):
            if line:
                splitted = line.split(": ")
                if len(splitted) == 2:
                    key, value = splitted
                    res[key] = value
        return res
    except subprocess.CalledProcessError:
        return res


def value_else_none(value):
    """Convert NaN to None"""
    if value != value:  # pylint: disable=comparison-with-itself
        return None
    return value


def get_device(force_cpu=False, device_type="auto"):
    """Get the best available device"""
    if force_cpu or device_type == "cpu":
        return "cpu"

    if device_type == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("GPU requested but CUDA is not available")

    # Auto mode - detect best available
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    return "cpu"


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DataCollator:
    """Custom data collator to handle padding while preserving label masking"""

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


def causal_lm_loss(logits, labels):
    """Causal language modeling loss function"""
    if labels is None:
        return torch.tensor(0.0, device=logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def evaluate_perplexity(model, dataloader, tokenizer, device, prompt=None):
    """Evaluate model perplexity on a dataset"""
    model.eval()
    model.to(device)
    total_loss, total_tokens = 0.0, 0

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

    # Generate response if prompt provided
    response = None
    if prompt:
        response = generate_and_print(prompt, model, tokenizer, seed=0)

    return perplexity, response


def run_benchmark(args):
    """Run the LLama LoRA fine-tuning benchmark"""
    # Set seed for reproducibility
    set_seed(args.seed)
    device = get_device(args.force_cpu, args.device_type)

    print(f"Device: {device}")
    print(f"Mode: {args.mode}, n_bits: {args.n_bits if args.mode != 'torch' else 'N/A'}")
    print(f"Max length: {args.max_length}, Batch size: {args.batch_size}")
    print(f"Training steps: {args.training_steps}, Eval interval: {args.eval_interval}")

    # Initialize timing
    start_time = time.time()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Load dataset
    dataset = load_dataset("microsoft/orca-math-word-problems-200k")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Filter by length
    def length_filter(example):
        return len(tokenizer.encode(example["question"])) <= args.max_length

    train_dataset = train_dataset.filter(length_filter)
    test_dataset = test_dataset.filter(length_filter)

    # Process examples
    def process_example(example):
        inputs = tokenizer(
            example["question"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    train_dataset = train_dataset.map(process_example, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(process_example, remove_columns=test_dataset.column_names)

    # Create data loaders
    # train_dataloader is not used, so we remove its assignment
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollator(tokenizer),
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        fp16=device == "cuda",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(tokenizer),
    )

    # Train model
    trainer.train()

    # Evaluate model
    perplexity, response = evaluate_perplexity(
        model, test_dataloader, tokenizer, device, args.eval_prompt
    )

    # Save model
    if args.save_model:
        model_path = Path(args.output_dir) / f"llama_lora_finetuned_{args.mode}"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    # Create results dictionary
    results = {
        "perplexity": perplexity,
        "response": response,
        "training_time": time.time() - start_time,
        "device": device,
        "mode": args.mode,
        "n_bits": args.n_bits if args.mode != "torch" else None,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "training_steps": args.training_steps,
        "eval_interval": args.eval_interval,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "system_info": get_system_information(),
        "ec2_metadata": get_ec2_metadata(),
    }

    return results


def create_benchmark_json(results):
    """Create the JSON output in the format expected by the benchmark database"""
    # Get system information
    system_info = get_system_information()
    ec2_metadata = get_ec2_metadata()

    # Get git information
    git_info = {"hash": "unknown", "timestamp": datetime.datetime.now().timestamp()}
    if CONVERT_AVAILABLE:
        path_to_repository = Path(__file__).parent.parent
        try:
            assert not is_git_diff(path_to_repository)
            git_hash = get_git_hash(path_to_repository)
            git_date = get_git_hash_date(git_hash, path_to_repository)
            git_timestamp = datetime.datetime.fromisoformat(
                git_iso_to_python_iso(git_date)
            ).timestamp()
            git_info = {"hash": git_hash, "timestamp": git_timestamp}
        except (subprocess.CalledProcessError, ValueError, AssertionError):
            pass

    # Build experiment name
    config = results["config"]
    experiment_name = f"llama-lora-math-{config['fhe_mode']}"
    if config["mode"] != "torch":
        experiment_name += f"-{config['n_bits']}bit"
    else:
        experiment_name += "-torch"
    experiment_name += f"-{config['device']}"

    # Build metrics
    metrics = []

    # Add timing metrics
    for key, value in results["timing_results"].items():
        metrics.append({"metric_name": key, "value": value_else_none(value)})

    # Add configuration metrics
    for key in ["batch_size", "max_length", "n_bits", "lora_r", "lora_alpha", "training_steps"]:
        if key in config:
            metrics.append({"metric_name": key, "value": value_else_none(config[key])})

    # Add perplexity metrics
    for key, value in results["metrics"].items():
        metrics.append({"metric_name": key, "value": value_else_none(value)})

    # Add device metrics
    metrics.append({"metric_name": "device_type", "value": config["device"]})

    # Calculate throughput
    if results["timing_results"]["training_time"] > 0:
        tokens_per_second = (
            config["batch_size"]
            * config["max_length"]
            * config["training_steps"]
            / results["timing_results"]["training_time"]
        )
        metrics.append(
            {"metric_name": "tokens_per_second", "value": value_else_none(tokens_per_second)}
        )

    # Build the complete structure
    session_data = {
        "machine": {
            "machine_name": ec2_metadata.get("instance-type", socket.gethostname()),
            "machine_specs": system_info,
        },
        "experiments": [
            {
                "experiment_name": experiment_name,
                "experiment_metadata": {
                    **config,
                    "model_architecture": "llama-3.2-1b",
                    "task": "lora-finetuning-math",
                    "dataset": "orca-math-word-problems-200k",
                    "framework": "concrete-ml",
                },
                "git_hash": git_info["hash"],
                "git_timestamp": git_info["timestamp"],
                "experiment_timestamp": datetime.datetime.now().timestamp(),
                "metrics": metrics,
            }
        ],
    }

    return session_data


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLama LoRA Fine-tuning Benchmark")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--mode", type=str, default="7bit", choices=["torch", "7bit", "16bit"])
    parser.add_argument("--n-bits", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-prompt", type=str, default="What is 2+2?")
    parser.add_argument("--output-dir", type=str, default="deployment")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--device-type", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmark
    results = run_benchmark(args)

    # Save results
    with open("to_upload.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
