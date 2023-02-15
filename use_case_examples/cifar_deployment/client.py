"""Client script.

This script does the following:
    - Query crypto-parameters and pre/post-processing parameters
    - Quantize the inputs using the parameters
    - Encrypt data using the crypto-parameters
    - Send the encrypted data to the server (async using grequests)
    - Collect the data and decrypt it
    - De-quantize the decrypted results
"""
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import List

import grequests
import numpy
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from concrete.numpy import Configuration
from model import CNV
from tqdm import tqdm

from concrete.ml.deployment import FHEModelClient
from concrete.ml.torch.compile import compile_brevitas_qat_model

URL = os.environ.get("URL", f"http://localhost:5000")
STATUS_OK = 200
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 1))


def main():
    model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    loaded = torch.load(Path(__file__).parent / "8_bit_model.pt")
    model.load_state_dict(loaded["model_state_dict"])
    model = model.eval()
    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    try:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=False,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    except:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=True,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )

    num_samples = 1000
    train_sub_set = torch.stack(
        [train_set[index][0] for index in range(min(num_samples, len(train_set)))]
    )

    # Pre-processing -> images -> feature maps
    with torch.no_grad():
        train_features_sub_set = model.clear_module(train_sub_set)

    # Get the necessary data for the client
    # client.zip
    zip_response = requests.get(f"{URL}/get_client")
    assert zip_response.status_code == STATUS_OK
    with open("./client.zip", "wb") as file:
        file.write(zip_response.content)

    # serialized_processing.json
    zip_response = requests.get(f"{URL}/get_processing")
    assert zip_response.status_code == STATUS_OK
    with open("./serialized_processing.json", "wb") as file:
        file.write(zip_response.content)

    # Get the data to infer
    X = train_features_sub_set[:1]

    # Let's create the client
    client = FHEModelClient(path_dir="./", key_dir="./keys")

    # The client first need to create the private and evaluation keys.
    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    # Evaluation keys can be quite large files but only have to be shared once with the server.

    # Check the size of the evaluation keys (in MB)
    print(f"Evaluation keys size: {sys.getsizeof(serialized_evaluation_keys) / 1024 / 1024:.2f} MB")

    # Update all base64 queries encodings with UploadFile
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2932
    response = requests.post(
        f"{URL}/add_key",
        files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)},
    )
    assert response.status_code == STATUS_OK
    uid = response.json()["uid"]

    inferences = []
    # Launch the queries
    clear_input = X[[0], :].numpy()
    print("Input shape:", clear_input.shape)

    assert isinstance(clear_input, numpy.ndarray)
    print("Quantize/Encrypt")
    encrypted_input = client.quantize_encrypt_serialize(clear_input)
    assert isinstance(encrypted_input, bytes)

    print(f"Encrypted input size: {sys.getsizeof(encrypted_input) / 1024 / 1024:.2f} MB")

    print("Posting query")
    inferences.append(
        grequests.post(
            f"{URL}/compute",
            files={
                "model_input": io.BytesIO(encrypted_input),
            },
            data={
                "uid": uid,
            },
        )
    )

    del encrypted_input
    del serialized_evaluation_keys

    print("Posted!")

    # Unpack the results
    decrypted_predictions = []
    for result in grequests.map(inferences):
        assert result.status_code == STATUS_OK
        print("OK!")

        encrypted_result = result.content
        decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
        decrypted_predictions.append(decrypted_prediction)
    print(decrypted_predictions)


if __name__ == "__main__":
    main()
