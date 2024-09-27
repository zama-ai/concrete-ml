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
from pathlib import Path

import grequests
import numpy
import requests
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm

from concrete import fhe
from concrete.ml.deployment import FHEModelClient

URL = os.environ.get("URL", f"http://localhost:8888")
STATUS_OK = 200
ROOT = Path(__file__).parent / "client"
ROOT.mkdir(exist_ok=True)

encrypt_with_tfhe = False
nb_samples = 10

def to_tuple(x) -> tuple:
    """Make the input a tuple if it is not already the case.

    Args:
        x (Any): The input to consider. It can already be an input.

    Returns:
        tuple: The input as a tuple.
    """
    # If the input is not a tuple, return a tuple of a single element
    if not isinstance(x, tuple):
        return (x,)

    return x

def serialize_encrypted_values(
    *values_enc,
):
    """Serialize encrypted values.

    If a value is None, None is returned.

    Args:
        values_enc (Optional[fhe.Value]): The values to serialize.

    Returns:
        Union[Optional[bytes], Optional[Tuple[bytes]]]: The serialized values.
    """
    values_enc_serialized = tuple(
        value_enc.serialize() if value_enc is not None else None for value_enc in values_enc
    )

    if len(values_enc_serialized) == 1:
        return values_enc_serialized[0]

    return values_enc_serialized

def deserialize_encrypted_values(
    *values_serialized,
):
    """Deserialize encrypted values.

    If a value is None, None is returned.

    Args:
        values_serialized (Optional[bytes]): The values to deserialize.

    Returns:
        Union[Optional[fhe.Value], Optional[Tuple[fhe.Value]]]: The deserialized values.
    """
    values_enc = tuple(
        fhe.Value.deserialize(value_serialized) if value_serialized is not None else None
        for value_serialized in values_serialized
    )

    if len(values_enc) == 1:
        return values_enc[0]

    return values_enc


if __name__ == "__main__":
    # Get the necessary data for the client
    # client.zip
    zip_response = requests.get(f"{URL}/get_client")
    assert zip_response.status_code == STATUS_OK
    with open(ROOT / "client.zip", "wb") as file:
        file.write(zip_response.content)

    # Get the data to infer
    X, y = load_breast_cancer(return_X_y=True)
    assert isinstance(X, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)
    X = X[-nb_samples:]
    y = y[-nb_samples:]

    assert isinstance(X, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)

    # Create the client
    client = FHEModelClient(path_dir=str(ROOT.resolve()), key_dir=str((ROOT / "keys").resolve()))

    # The client first need to create the private and evaluation keys.
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()

    assert isinstance(serialized_evaluation_keys, bytes)

    # Evaluation keys can be quite large files but only have to be shared once with the server.

    # Check the size of the evaluation keys (in MB)
    print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")

    # Send this evaluation key to the server (this has to be done only once)
    # send_evaluation_key_to_server(serialized_evaluation_keys)

    # Now we have everything for the client to interact with the server

    # We create a loop to send the input to the server and receive the encrypted prediction
    execution_time = []
    encrypted_input = None
    clear_input = None

    # Update all base64 queries encodings with UploadFile
    response = requests.post(
        f"{URL}/add_key", files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)}
    )
    assert response.status_code == STATUS_OK
    uid = response.json()["uid"]

    inferences = []
    # Launch the queries
    for i in tqdm(range(len(X))):
        clear_input = X[[i], :]

        assert isinstance(clear_input, numpy.ndarray)

        quantized_input = to_tuple(client.model.quantize_input(clear_input))

        # Here, we can encrypt with TFHE-rs instead of Concrete
        if encrypt_with_tfhe:
            pass
        else:
            encrypted_input = to_tuple(client.client.encrypt(*quantized_input))

        encrypted_input = serialize_encrypted_values(*encrypted_input)

        # Debugging
        if False:
            print(f"Clear input: {clear_input}")
            print(f"Quantized input: {quantized_input}")
            print(f"Quantized input: {encrypted_input}")

        assert isinstance(encrypted_input, bytes)

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

    # Unpack the results
    decrypted_predictions = []
    for result in grequests.map(inferences):
        if result is None:
            raise ValueError("Result is None, probably due to a crash on the server side.")
        assert result.status_code == STATUS_OK

        encrypted_result = result.content

        # Decrypt and deserialize the values
        result_quant_encrypted = to_tuple(
            deserialize_encrypted_values(encrypted_result)
        )

        result_quant = to_tuple(client.client.decrypt(*result_quant_encrypted))

        result = to_tuple(client.model.dequantize_output(*result_quant))
        decrypted_prediction = client.model.post_processing(*result)[0]

        decrypted_predictions.append(decrypted_prediction)
    print(f"Decrypted predictions are: {decrypted_predictions}")

    decrypted_predictions_classes = numpy.array(decrypted_predictions).argmax(axis=1)
    print(f"Decrypted prediction classes are: {decrypted_predictions_classes}")

    # Let's check the results and compare them against the clear model
    clear_prediction_classes = y[0:nb_samples]
    accuracy = (clear_prediction_classes == decrypted_predictions_classes).mean()
    print(f"Accuracy between FHE prediction and expected results is: {accuracy*100:.0f}%")

