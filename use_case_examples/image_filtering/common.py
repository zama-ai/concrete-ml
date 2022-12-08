"All the constants used in this repo."

from pathlib import Path
from PIL import Image
import numpy as np

# The repo's directory
REPO_DIR = Path(__file__).parent

# The repo's main directories 
FILTERS_PATH = REPO_DIR / "filters"
KEYS_PATH = REPO_DIR / ".fhe_keys"
CLIENT_TMP_PATH = REPO_DIR / "client_tmp"
SERVER_TMP_PATH = REPO_DIR / "server_tmp"

# Create the directories if it does not exist yet
KEYS_PATH.mkdir(exist_ok=True)
CLIENT_TMP_PATH.mkdir(exist_ok=True)
SERVER_TMP_PATH.mkdir(exist_ok=True)

# All the filters currently available in the app 
AVAILABLE_FILTERS = [
    "identity",
    "inverted",
    "rotate",
    "black and white",
    "blur",
    "sharpen",
    "ridge detection",
]

# The input image's shape. Images with larger input shapes will be cropped and/or resized to this
INPUT_SHAPE = (100, 100)

# Generate random images as an inpuset for compilation
np.random.seed(42)
INPUTSET = tuple(
    np.random.randint(0, 255, size=(INPUT_SHAPE + (3,)), dtype=np.int64) for _ in range(10)
)

def load_image(image_path):
    image = Image.open(image_path).convert('RGB').resize(INPUT_SHAPE)
    image = np.asarray(image, dtype="int64")
    return image

_INPUTSET_DIR = REPO_DIR / "input_examples"

# List of all image examples suggested in the app
EXAMPLES = [str(image) for image in _INPUTSET_DIR.glob('**/*')]

SERVER_URL = "http://localhost:8000/"
