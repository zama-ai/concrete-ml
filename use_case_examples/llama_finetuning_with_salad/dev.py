import subprocess
import sys
import os
import time
import signal
from datasets import load_from_disk
from tqdm import tqdm
import math
from copy import deepcopy
from pathlib import Path
from utils_dev import *
import torch
from concrete.ml.torch.lora import get_remote_names
from concrete.ml.torch.hybrid_model import HybridFHEModel
from concrete.ml.torch.hybrid_model import HybridFHEMode

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from torch.utils.data import DataLoader

from concrete.ml.torch.lora import LoraTraining
from concrete.ml.torch.lora import LoraTrainer

from peft import LoraConfig
from peft import get_peft_model


from datasets import Dataset
from utils_lora import generate_and_print

from concrete.ml.torch.hybrid_model import  HybridFHEModel, HybridFHEModelServer, FHEModelServer, tuple_to_underscore_str
from concrete.ml.torch.hybrid_model import HybridFHEModelServer, HybridFHEMode, underscore_str_to_tuple

from utils_dev import *

# On dev Side, we know:
# which model will be finetuned
# the finetuning params
# the shape of the data


peft_args = {'r': 8,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': "none",
               'task_type': "CAUSAL_LM",
               'target_modules': "all-linear",
               }

lora_args = None

training_args = {
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


DEVICE = get_device(force_device='cpu')

if __name__ == "__main__":

    purge_compiled_model_dir(COMPILED_MODELS_PAH, delete=True)

    ########### Load data-set
    print(f'Load Data...')
    collator = DataCollator(TOKENIZER)

    train_dataset = load_from_disk(TRAIN_PATH)
    test_dataset = load_from_disk(TEST_PATH)

    ########### Load pre-trained model
    print(f'Load pre-trained model...')
    pretrained_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    pretrained_model.config.pad_token_id = pretrained_model.config.eos_token_id

    # Freeze model weights
    if FREEZE_WEIGTHS:
        # Freeze all model parameters
        for param in pretrained_model.parameters():
            param.requires_grad = False

    ########## Inject PEFT features
    # Injecting specific modules to fine-tune a pre-entrainer model
    # while considerably reducing the number of parameters to be trained

    print(f'Inject PEFT features...')
    peft_model = get_peft_model(pretrained_model, LoraConfig(**peft_args))
    peft_model.to(DEVICE)

    # peft_model.save_pretrained(f"{MODEL_DIR}/saved_peft_model/")
    # TOKENIZER.save_pretrained(f"{MODEL_DIR}/saved_tokenizer/")


    ########## Inject LORA trainer features
    # Injecting specific modules to train a pre-entrainer model using LORQ

    print(f'Inject LORA trainer features...')
    from transformers import Trainer

    hf_trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(**training_args),
        train_dataset=train_dataset,
        data_collator=collator,
    )

    train_dl = hf_trainer.get_train_dataloader()
    eval_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    hf_trainer.create_optimizer_and_scheduler(len(train_dl) * training_args['num_train_epochs'])
    optimizer, lr_scheduler = hf_trainer.optimizer, hf_trainer.lr_scheduler

    lora_trainer = LoraTrainer(
        model=peft_model,
        optimizer=optimizer,
        loss_fn=causal_lm_loss,
        lr_scheduler=lr_scheduler,
        training_args=training_args,
        n_layers_to_skip_for_backprop=3,
        eval_loader=eval_dl,
        eval_metric_fn=metric_fn,
        logging_steps=1,
        eval_steps=100,
        train_log_path=TRAIN_LOG_FILE,
        optimized_linear_execution=False,
        server_remote_address="http://0.0.0.0:8000",
        model_name=f"meta-llama",
        verbose=True,
    )

    ########## Compilation

    print('Compilation ...')

    lora_trainer.compile(get_random_inputset(vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH), n_bits=N_BITS)

    print('Saving models...')

    lora_trainer.save_and_clear_private_info(MODEL_DIR, via_mlir=True)
    peft_model.save_pretrained(f"{COMPILED_MODELS_PAH}/artefact")
    pretrained_model.config.save_pretrained(f"{COMPILED_MODELS_PAH}/artefact")

    # artefact/
    # ├── adapter_config.json    ← config PEFT
    # ├── adapter_model.bin      ← LoRA weights
    # ├── config.json            ← (pad_token_id)

    print('<!> Now run server.py...')

