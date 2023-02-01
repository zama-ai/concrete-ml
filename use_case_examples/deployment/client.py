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
import os
import sys

import grequests
import numpy
import requests
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm

from concrete.ml.deployment import FHEModelClient

URL = os.environ.get("URL", f"http://localhost:5000")
STATUS_OK = 200

if __name__ == "__main__":
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
    X, y = load_breast_cancer(return_X_y=True)
    assert isinstance(X, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)
    X = X[-10:]
    y = y[-10:]

    assert isinstance(X, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)

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

    # Let's send this evaluation key to the server (this has to be done only once)
    # send_evaluation_key_to_server(serialized_evaluation_keys)

    # Now we have everything for the client to interact with the server

    # We create a loop to send the input to the server and receive the encrypted prediction
    execution_time = []
    encrypted_input = None
    clear_input = None

    # Update all base64 queries encodings with UploadFile
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2932
    response = requests.post(
        f"{URL}/add_key",
        json={"key": base64.b64encode(serialized_evaluation_keys).decode("ascii")},
    )
    assert response.status_code == STATUS_OK
    uid = response.json()["uid"]

    inferences = []
    # Launch the queries
    for i in tqdm(range(len(X))):
        clear_input = X[[i], :]

        assert isinstance(clear_input, numpy.ndarray)
        encrypted_input = client.quantize_encrypt_serialize(clear_input)
        assert isinstance(encrypted_input, bytes)

        inferences.append(
            grequests.post(
                f"{URL}/compute",
                json={"uid": uid, "inputs": base64.b64encode(encrypted_input).decode("ascii")},
            )
        )

    # Unpack the results
    decrypted_predictions = []
    for result in grequests.map(inferences):
        assert result.status_code == STATUS_OK
        result = result.json()["result"]
        assert isinstance(result, str)
        decrypted_prediction = client.deserialize_decrypt_dequantize(base64.b64decode(result))[0]
        decrypted_predictions.append(decrypted_prediction)
    print(decrypted_predictions)
