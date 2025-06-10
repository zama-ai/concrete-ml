## Common

from datasets import load_from_disk
from tqdm import tqdm
import math
from copy import deepcopy
from pathlib import Path
from utils_dev import *
import torch
from concrete.ml.torch.lora import get_remote_names
from concrete.ml.torch.hybrid_model import HybridFHEModel

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

N_BITS = 7
BATCH_SIZE = 4
MODE = f"{N_BITS}bit"
FREEZE_WEIGTHS = True
TRAIN_LOG_FILE = f"training_log_{MODE}.txt"
EVAL_RESPONSES_FILE = f"eval_generated_responses_{MODE}.txt"
PROMPT = "When you multiply a number by 7, it becomes 98. What is that number?\n"

DEVICE = get_device(force_device='cpu')

MODEL_DIR = COMPILED_MODELS_PAH / MODEL_NAME

if __name__ == "__main__":

    purge_compiled_model_dir(COMPILED_MODELS_PAH)

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

    ########## Inject LORA trainer features
    # Injecting specific modules to train a pre-entrainer model using LORQ

    print(f'Inject LORA trainer features...')
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
    )

    ########## Compilation

    print('Compilation ...')
    lora_trainer.compile(get_random_inputset(vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE, max_length=MAX_LENGTH), n_bits=N_BITS)

    lora_trainer.save_and_clear_private_info(MODEL_DIR, via_mlir=True)

    # TODO <!>
    # Compile the model for multiple input shapes (i.e., varying context lengths)
    # because during generation, the model processes input sequences token by token.
    # As a result, intermediate activations (e.g., hidden states) can have different shapes,
    # To support this dynamic behavior with FHE execution, we need to compile and store
    # FHE clients for each possible shape that may occur during the generation loop.

    # Generate random inputset

    # Compile for different shapes
#     for length in range(6, MAX_LENGTH):
#         inputset = get_random_inputset(
#             vocab_size=vocab_size,
#             batch_size=BATCH_SIZE,
#             max_length=length
#         )
#         print(f"Compiling for block_size={length}, input shape: {inputset['input_ids'].shape}")
#         print(f'{inputset.keys()=}, {inputset["input_ids"].shape}')

#         lora_trainer = LoraTrainer(
#             model=deepcopy(peft_model),
#             optimizer=optimizer,
#             loss_fn=causal_lm_loss,
#             lr_scheduler=lr_scheduler,
#             training_args=training_args,
#             n_layers_to_skip_for_backprop=3,
#             eval_loader=eval_dl,
#             eval_metric_fn=metric_fn,
#             logging_steps=1,
#             eval_steps=10,
#             train_log_path=TRAIN_LOG_FILE,
#             optimized_linear_execution=False,
#             server_remote_address="http://0.0.0.0:8000",
#             model_name=f"{MODEL_NAME}",
#             verbose=True,
#             remote_names[:1]
#         )

#         # Get the names of the remote modules (layers to be converted to FHE)
#         if length == 0:
#             print(f"`{len(lora_trainer.remote_names)}` remote modules") # 221 layers

#         lora_trainer.compile(inputset, n_bits=N_BITS, use_dynamic_quantization=True)

#         # (bool): if fhe circuits should be serialized using via_mlir option useful
#         # for cross-platform (compile on one architecture and run on another)
#         lora_trainer.save_and_clear_private_info(model_dir, via_mlir=True)



# # /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
# #   warnings.warn('resource_tracker: There appear to be %d '
# # (.lora-venv) kcelia@MacBook-Pro-de-Celia lora_finetuning %
