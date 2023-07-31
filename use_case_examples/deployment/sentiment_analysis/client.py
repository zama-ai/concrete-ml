#! /bin/env python
"""Client script.

This script does the following:
    - Query crypto-parameters and pre/post-processing parameters
    - Quantize the inputs using the parameters
    - Encrypt data using the crypto-parameters
    - Send the encrypted data to the server (async using grequests)
    - Collect the data and decrypt it
    - De-quantize the decrypted results
"""
import io
import os
import time

os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
import os
import sys
import time

import grequests
import numpy
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utility_functions import text_to_tensor

from concrete.ml.deployment import FHEModelClient

URL = os.environ.get("URL", f"http://localhost:5000")
STATUS_OK = 200

CLASS_INDEX_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    # Load the tokenizer (converts text to tokens)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Load the pre-trained model
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    # Get the necessary data for the client
    # client.zip
    zip_response = requests.get(f"{URL}/get_client")
    assert zip_response.status_code == STATUS_OK
    with open("./client.zip", "wb") as file:
        file.write(zip_response.content)

    # Create the client
    client = FHEModelClient(path_dir="./", key_dir="./keys")

    # The client first need to create the private and evaluation keys.
    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)
    # Evaluation keys can be quite large files but only have to be shared once with the server.
    # Check the size of the evaluation keys (in MB)
    print(f"Evaluation keys size: {sys.getsizeof(serialized_evaluation_keys) / 1024 / 1024:.2f} MB")

    response = requests.post(
        f"{URL}/add_key",
        files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)},
    )
    del serialized_evaluation_keys
    assert response.status_code == STATUS_OK
    uid = response.json()["uid"]

    while True:
        inputs = input("Text to classify: ")
        # Launch the queries
        print("extracting feature vector")
        clear_input = text_to_tensor(
            [inputs], transformer_model=transformer_model, tokenizer=tokenizer, device="cpu"
        )
        print("Input shape:", clear_input.shape)

        start = time.time()
        assert isinstance(clear_input, numpy.ndarray)
        print("Quantize/Encrypt")
        encrypted_input = client.quantize_encrypt_serialize(clear_input)
        assert isinstance(encrypted_input, bytes)
        print(f"Encrypted input size: {sys.getsizeof(encrypted_input) / 1024 / 1024:.2f} MB")

        print("Posting query ...")
        inferences = [
            grequests.post(
                f"{URL}/compute",
                files={
                    "model_input": io.BytesIO(encrypted_input),
                },
                data={
                    "uid": uid,
                },
            )
        ]
        del encrypted_input
        print("Posted!")

        # Unpack the results
        result = grequests.map(inferences)[0]
        if result is None:
            raise ValueError("Result is None, probably due to a crash on the server side.")
        assert result.status_code == STATUS_OK
        print("OK!")

        encrypted_result = result.content
        decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)
        end = time.time()

        predicted_class = numpy.argmax(decrypted_prediction)
        print(
            f"This tweet is {CLASS_INDEX_TO_NAME[predicted_class]}!\n"
            f"Probabilites={decrypted_prediction.tolist()}\n"
            f"It took {end - start} seconds to run."
        )


if __name__ == "__main__":
    main()
