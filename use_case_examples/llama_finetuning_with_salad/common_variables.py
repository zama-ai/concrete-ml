
from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.2-1B"
COMPILED_MODELS_PAH = Path("compiled_models")

PATH_TO_CLIENTS = COMPILED_MODELS_PAH / f"{MODEL_NAME}"
PATH_TO_CLIENTS_KEYS = COMPILED_MODELS_PAH / f"{MODEL_NAME}_keys"

MAX_LENGTH = 64

N_BITS = 7
BATCH_SIZE = 4
MODE = f"{N_BITS}bit"
FREEZE_WEIGTHS = True
TRAIN_LOG_FILE = f"training_log_{MODE}.txt"
EVAL_RESPONSES_FILE = f"eval_generated_responses_{MODE}.txt"
PROMPT = "When you multiply a number by 7, it becomes 98. What is that number?\n"

