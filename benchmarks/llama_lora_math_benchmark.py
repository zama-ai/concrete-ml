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
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import cpuinfo
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset
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

# Import from concrete.ml
from concrete.ml.torch.lora import LoraTrainer

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
        else:
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
    
    # Store original FHE mode and set to disable for evaluation
    original_fhe_mode = None
    if hasattr(model, "hybrid_model"):
        original_fhe_mode = model.hybrid_model.fhe_mode
        model.hybrid_model.set_fhe_mode("disable")
    
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
    
    # Restore original FHE mode if it was changed
    if original_fhe_mode is not None:
        model.hybrid_model.set_fhe_mode(original_fhe_mode)
    
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
    
    # Override use_cpu for GPU runs
    if device == "cuda":
        args.use_cpu = False
    
    print(f"Device: {device}")
    print(f"Mode: {args.mode}, n_bits: {args.n_bits if args.mode != 'torch' else 'N/A'}")
    print(f"Max length: {args.max_length}, Batch size: {args.batch_size}")
    print(f"Training steps: {args.training_steps}, Eval interval: {args.eval_interval}")

    # Initialize timing
    timing_results = {}
    total_start_time = time.time()

    # Setup phase
    setup_start_time = time.time()
    print("\n⏱️  Starting setup phase...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Freeze the original model's weights
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    peft_model = get_peft_model(model, peft_config)

    # Load and preprocess dataset
    print("Loading dataset...")
    if args.debug_mode:
        # In debug mode, load a small subset of the dataset
        raw_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:100]")
        print("Debug mode: Using only 100 examples from the dataset")
    else:
        raw_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    
    def length_filter(example):
        q_len = len(tokenizer(example["question"], add_special_tokens=False)["input_ids"])
        a_len = len(tokenizer(example["answer"], add_special_tokens=False)["input_ids"])
        return (q_len + a_len + 1) <= args.max_length

    filtered_dataset = raw_dataset.filter(length_filter)
    print(f"Filtered dataset size: {len(filtered_dataset):,} (from {len(raw_dataset):,})")

    def process_example(example):
        """Tokenize a question-answer pair and prepare labels for training"""
        question = example["question"].strip()
        answer = example["answer"].strip()
        tokens = tokenizer(
            question + "\n" + answer,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        question_length = len(tokenizer(question, add_special_tokens=False)["input_ids"]) + 1
        labels = tokens["input_ids"].copy()
        for i in range(question_length):
            if i < len(labels):
                labels[i] = -100
        tokens["labels"] = labels
        return tokens

    tokenized_dataset = filtered_dataset.map(
        process_example,
        batched=False,
        remove_columns=filtered_dataset.column_names,
    )

    # Split dataset
    tokenized = tokenized_dataset.train_test_split(test_size=0.05, seed=args.seed, shuffle=True)
    train_dataset, test_dataset = tokenized["train"], tokenized["test"]
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Create data collator
    collator = DataCollator(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        save_total_limit=1,
        use_cpu=(device == "cpu"),
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        seed=args.seed,
        data_seed=args.seed,
        warmup_steps=10,
        weight_decay=0.01,
        prediction_loss_only=True,
        report_to="none",
    )

    # Create optimizer and scheduler
    hf_trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataloader = hf_trainer.get_train_dataloader()
    hf_trainer.create_optimizer_and_scheduler(len(train_dataloader) * training_args.num_train_epochs)

    optimizer = hf_trainer.optimizer
    lr_scheduler = hf_trainer.lr_scheduler

    setup_end_time = time.time()
    timing_results["setup_time"] = setup_end_time - setup_start_time
    print(f"✅ Setup completed in {timing_results['setup_time']:.2f} seconds")

    # Compilation phase (if not torch mode)
    if args.mode != "torch":
        compilation_start_time = time.time()
        print(f"\n⏱️  Starting compilation phase (n_bits={args.n_bits})...")

        # Use actual data from the dataset for calibration
        calibration_batch = next(iter(train_dataloader))
        inputset = {
            "input_ids": calibration_batch["input_ids"],
            "attention_mask": calibration_batch["attention_mask"],
            "labels": calibration_batch["labels"]
        }

        # Initialize LoraTrainer
        lora_trainer = LoraTrainer(
            model=peft_model,
            optimizer=optimizer,
            loss_fn=causal_lm_loss,
            lr_scheduler=lr_scheduler,
            training_args=vars(training_args),
            n_layers_to_skip_for_backprop=args.n_layers_to_skip,
        )

        if args.debug_mode:
            print(f"Debug mode using only the first layer in FHE: {lora_trainer.hybrid_model.module_names[0]}")
            lora_trainer.hybrid_model.module_names = lora_trainer.hybrid_model.module_names[:1]
            selected_module_name = lora_trainer.hybrid_model.module_names[0]
            # Remove all other remote modules except the selected one
            for name, module in list(lora_trainer.hybrid_model.remote_modules.items()):
                if name != selected_module_name:
                    # Replace the remote module with its private_module in the model
                    *path, last = name.split(".")
                    parent_module = lora_trainer.hybrid_model._get_module_by_name(lora_trainer.hybrid_model.model, ".".join(path)) if path else lora_trainer.hybrid_model.model
                    setattr(parent_module, last, module.private_module)
            # Keep only the selected remote module in remote_modules
            lora_trainer.hybrid_model.remote_modules = {selected_module_name: lora_trainer.hybrid_model.remote_modules[selected_module_name]}

            # Set FHE mode to CALIBRATE to ensure proper calibration data collection
            lora_trainer.hybrid_model.set_fhe_mode("calibrate")

            # Run forward passes over all batches in train_dataloader to collect calibration data
            lora_trainer.hybrid_model.model.eval()
            with torch.no_grad():
                try:
                    for batch in train_dataloader:
                        lora_trainer.hybrid_model.model((batch["input_ids"], batch["labels"]))
                except Exception as e:
                    print(f"Warning: Calibration forward pass failed: {e}")
                    print("Continuing with compilation...")

        # Compile the model
        lora_trainer.compile(inputset, n_bits=args.n_bits)

        compilation_end_time = time.time()
        timing_results["compilation_time"] = compilation_end_time - compilation_start_time
        print(f"✅ Compilation completed in {timing_results['compilation_time']:.2f} seconds")
    else:
        # For torch mode, no compilation needed
        timing_results["compilation_time"] = 0.0
        lora_trainer = LoraTrainer(
            model=peft_model,
            optimizer=optimizer,
            loss_fn=causal_lm_loss,
            lr_scheduler=lr_scheduler,
            training_args=vars(training_args),
            n_layers_to_skip_for_backprop=args.n_layers_to_skip,
        )

    # Evaluation setup
    eval_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collator)
    
    # Test prompt for generation
    PROMPT = "When you multiply a number by 7, it becomes 98. What is that number?\n"
    
    # Pre-training evaluation
    print("\n⏱️  Evaluating pre-training model...")
    eval_start = time.time()
    pre_perplexity, pre_response = evaluate_perplexity(peft_model, eval_dataloader, tokenizer, device, PROMPT)
    timing_results["pre_eval_time"] = time.time() - eval_start
    print(f"Pre-training perplexity: {pre_perplexity:.2f}")

    # Training phase
    training_start_time = time.time()
    print(f"\n⏱️  Starting training phase (fhe_mode={args.fhe_mode}, steps={args.training_steps})...")

    # Get limited batches for training
    train_batches = []
    for i, batch in enumerate(train_dataloader):
        if i < args.training_steps:
            train_batches.append(batch)
        else:
            break

    # Run training
    lora_trainer.train(train_batches, fhe=args.fhe_mode, device=device)

    training_end_time = time.time()
    timing_results["training_time"] = training_end_time - training_start_time
    timing_results["training_time_per_step"] = timing_results["training_time"] / len(train_batches)
    print(f"✅ Training completed in {timing_results['training_time']:.2f} seconds")

    # Post-training evaluation
    print("\n⏱️  Evaluating post-training model...")
    eval_start = time.time()
    post_perplexity, post_response = evaluate_perplexity(peft_model, eval_dataloader, tokenizer, device, PROMPT)
    timing_results["post_eval_time"] = time.time() - eval_start
    print(f"Post-training perplexity: {post_perplexity:.2f}")
    print(f"Perplexity improvement: {pre_perplexity - post_perplexity:.2f}")

    # Save model if requested
    if args.save_model:
        save_path = Path(f"deployment/llama_lora_finetuned_{args.mode}")
        if save_path.is_dir() and any(save_path.iterdir()):
            shutil.rmtree(save_path)
        lora_trainer.save_and_clear_private_info(save_path)
        print(f"✅ Model saved to {save_path}")

    total_end_time = time.time()
    timing_results["total_time"] = total_end_time - total_start_time

    # Build results
    results = {
        "timing_results": timing_results,
        "config": {
            "model_name": args.model_name,
            "mode": args.mode,
            "fhe_mode": args.fhe_mode,
            "n_bits": args.n_bits,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "training_steps": args.training_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "device": device,
        },
        "metrics": {
            "pre_perplexity": pre_perplexity,
            "post_perplexity": post_perplexity,
            "perplexity_improvement": pre_perplexity - post_perplexity,
        },
        "generations": {
            "prompt": PROMPT,
            "pre_training": pre_response,
            "post_training": post_response,
        }
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
        except:
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
        metrics.append({
            "metric_name": key,
            "value": value_else_none(value)
        })
    
    # Add configuration metrics
    for key in ["batch_size", "max_length", "n_bits", "lora_r", "lora_alpha", "training_steps"]:
        if key in config:
            metrics.append({
                "metric_name": key,
                "value": value_else_none(config[key])
            })
    
    # Add perplexity metrics
    for key, value in results["metrics"].items():
        metrics.append({
            "metric_name": key,
            "value": value_else_none(value)
        })
    
    # Add device metrics
    metrics.append({
        "metric_name": "device_type",
        "value": config["device"]
    })
    
    # Calculate throughput
    if results["timing_results"]["training_time"] > 0:
        tokens_per_second = (
            config["batch_size"] * config["max_length"] * config["training_steps"]
            / results["timing_results"]["training_time"]
        )
        metrics.append({
            "metric_name": "tokens_per_second",
            "value": value_else_none(tokens_per_second)
        })

    # Build the complete structure
    session_data = {
        "machine": {
            "machine_name": ec2_metadata.get("instance-type", socket.gethostname()),
            "machine_specs": system_info,
        },
        "experiments": [{
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
            "metrics": metrics
        }]
    }

    return session_data


def main():
    parser = argparse.ArgumentParser(description="LLama LoRA Fine-tuning Benchmark for Math Word Problems")
    
    # Model configuration
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B", 
                        help="Model name from HuggingFace")
    
    # Training mode and FHE configuration
    parser.add_argument("--mode", default="7bit", 
                        choices=["torch", "7bit", "16bit"],
                        help="Training mode")
    parser.add_argument("--fhe-mode", default="disable", 
                        choices=["disable", "simulate", "execute"],
                        help="FHE execution mode")
    
    # Training configuration
    parser.add_argument("--max-length", type=int, default=64,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--training-steps", type=int, default=1,
                        help="Number of training steps to run")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Steps between evaluations")
    
    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--n-layers-to-skip", type=int, default=3,
                        help="Number of layers to skip for backprop")
    
    # Other options
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--device-type", choices=["cpu", "gpu", "auto"], default="auto",
                    help="Device type to use (cpu, gpu, or auto-detect)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the fine-tuned model")
    parser.add_argument("--output", default="to_upload.json",
                        help="Output JSON file path")
    parser.add_argument("--use-cpu", action="store_true", default=True,
                        help="Set use_cpu=True in TrainingArguments (required for Concrete ML LoraTrainer, unless you want to run on GPU)")
    parser.add_argument("--debug-mode", action="store_true",
                        help="Enable debug mode to force a single module to be compiled")
    
    args = parser.parse_args()

    # Set n_bits based on mode
    if args.mode == "7bit":
        args.n_bits = 7
    elif args.mode == "16bit":
        args.n_bits = 16
    else:
        args.n_bits = None

    # Override from environment variables if present
    if "MODE" in os.environ:
        args.mode = os.environ["MODE"]
        if args.mode == "7bit":
            args.n_bits = 7
        elif args.mode == "16bit":
            args.n_bits = 16
        else:
            args.n_bits = None
    if "FHE_MODE" in os.environ:
        args.fhe_mode = os.environ["FHE_MODE"]
    if "MAX_LENGTH" in os.environ:
        args.max_length = int(os.environ["MAX_LENGTH"])
    if "BATCH_SIZE" in os.environ:
        args.batch_size = int(os.environ["BATCH_SIZE"])
    if "TRAINING_STEPS" in os.environ:
        args.training_steps = int(os.environ["TRAINING_STEPS"])

    # Run the benchmark
    print("🚀 Starting LLama LoRA Fine-tuning Benchmark for Math Word Problems")
    results = run_benchmark(args)

    # Create and save the benchmark JSON
    benchmark_data = create_benchmark_json(results)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)
    
    print(f"\n✅ Benchmark completed! Results saved to {args.output}")
    print(f"Total time: {results['timing_results']['total_time']:.2f} seconds")
    print(f"Final perplexity: {results['metrics']['post_perplexity']:.2f}")


if __name__ == "__main__":
    main()