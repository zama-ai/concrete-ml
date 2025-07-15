import argparse
from time import time

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from utils_dev import *

from concrete.ml.torch.hybrid_model import HybridFHEMode
from concrete.ml.torch.lora import LoraTrainer

# On The dev Side, we know:
# - Which model will be finetuned
# - The finetuning params
# - The shape of the data

# ========================= LoRA / PEFT Config ==========================
PEFT_ARGS = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
}

TRAINING_ARGS = {
    "output_dir": "./checkpoints",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "save_total_limit": 1,
    "use_cpu": True,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "seed": SEED,
    "data_seed": SEED,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "prediction_loss_only": True,
    "report_to": "none",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LORA fine-tuning with FHE options")
    parser.add_argument("--save_compiled_model", default=True)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    DEVICE = get_device(force_device=args.device)

    purge_compiled_model_dir(COMPILED_MODELS_PATH, delete=args.save_compiled_model)
    print(f"--> Fine-tuning..")

    # --------------------- [1] Load Data ---------------------
    print(f"--> Load Data...")
    collator = DataCollator(TOKENIZER)
    train_dataset = load_from_disk(TRAIN_PATH)
    test_dataset = load_from_disk(TEST_PATH)

    # --------------------- [2] Load Pretrained Model ---------------------
    print(f"--> Load pre-trained model...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
    pretrained_model.config.pad_token_id = pretrained_model.config.eos_token_id

    if FREEZE_WEIGTHS:
        for param in pretrained_model.parameters():
            param.requires_grad = False

    # --------------------- [3] Inject LoRA ---------------------
    # Injecting specific modules to fine-tune a pre-entrainer model
    # while considerably reducing the number of parameters to be trained
    print(f"--> Inject PEFT features...")
    peft_model = get_peft_model(pretrained_model, LoraConfig(**PEFT_ARGS)).to(DEVICE)

    # --------------------- [4] HuggingFace Trainer ---------------------
    # Injecting specific modules to train a pre-entrainer model using LORQ
    print(f"--> Inject HF trainer features...")
    hf_trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(**TRAINING_ARGS),
        train_dataset=train_dataset,
        data_collator=collator,
    )

    train_dl = hf_trainer.get_train_dataloader()
    eval_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    hf_trainer.create_optimizer_and_scheduler(len(train_dl) * TRAINING_ARGS["num_train_epochs"])
    optimizer, lr_scheduler = hf_trainer.optimizer, hf_trainer.lr_scheduler

    # --------------------- [5] CML LoraTrainer ---------------------
    print("--> Inject CML LoraTrainer features...")
    lora_trainer = LoraTrainer(
        model=peft_model,
        optimizer=optimizer,
        loss_fn=causal_lm_loss,
        lr_scheduler=lr_scheduler,
        training_args=TRAINING_ARGS,
        n_layers_to_skip_for_backprop=3,
        eval_loader=eval_dl,
        eval_metric_fn=metric_fn,
        logging_steps=1,
        eval_steps=100,
        train_log_path=TRAIN_LOG_FILE,
        machine_type="M4",
        # server_remote_address="http://0.0.0.0:8000",
        server_remote_address="http://127.0.0.1:8001",
        # server_remote_address='https://mango-arugula-68eafimvf0z5o8ci.salad.cloud/',
        # server_remote_address="http://[::1]:8000",
        # server_remote_address="http://51.44.244.35:8000",
        model_name=f"meta-llama",
    )

    # --------------------- [6] Compilation ---------------------
    print(f"--> Compilation with {DEVICE=}...")
    inputset = get_random_inputset(
        vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, device=DEVICE
    )
    start_time = time()
    lora_trainer.compile(inputset, n_bits=N_BITS, device=DEVICE)
    print(f"Compilation completed under: {time() - start_time:.2f}s using {DEVICE=}")
    # Compilation completed under: 5.24s using DEVICE='cpu'

    if args.save_compiled_model:
        print(f"--> Saving compiled models at {COMPILED_MODELS_PATH=}...")
        lora_trainer.save_and_clear_private_info(COMPILED_MODELS_PATH, via_mlir=True)

    print("--> <!> Now run `server_<compilation_type>.py`...")
    # --------------------- [7] Init Client ---------------------
    print("--> Init FHE client...")
    client_path = COMPILED_MODELS_PATH / "client"
    lora_trainer.hybrid_model.init_client(
        path_to_clients=client_path, path_to_keys=PATH_TO_CLIENTS_KEYS
    )
    lora_trainer.hybrid_model.set_fhe_mode(HybridFHEMode.REMOTE)

    # --------------------- [8] Fine-tuning ---------------------
    print("--> Running FHE remote training...")
    limited_batches = get_limited_batches(train_dl, 1)
    lora_trainer.train(limited_batches, fhe="remote", device=DEVICE)
