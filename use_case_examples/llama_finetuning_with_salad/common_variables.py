
from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.2-1B"
COMPILED_MODELS_PAH = Path("compiled_models")

PATH_TO_CLIENTS = COMPILED_MODELS_PAH / f"{MODEL_NAME}"
PATH_TO_CLIENTS_KEYS = COMPILED_MODELS_PAH / f"{MODEL_NAME}_keys"

MAX_LENGTH = 64

