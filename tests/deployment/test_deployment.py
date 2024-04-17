"""Test deployment."""

import io
import time
import uuid
import warnings
from pathlib import Path

import numpy
import pytest
import requests
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.deployment import FHEModelClient, FHEModelDev
from concrete.ml.deployment.deploy_to_aws import AWSInstance, deploy_to_aws
from concrete.ml.deployment.utils import wait_for_connection_to_be_available
from concrete.ml.pytest.utils import instantiate_model_generic
from concrete.ml.sklearn.linear_model import LogisticRegression

STATUS_OK = 200
# Internal AWS AMI with private credentials
AMI_ID_WITH_CREDENTIALS = "ami-08fe791e2421787f1"


def test_timeout_ssh_connection():
    """Test timeout error raised on ssh connection"""
    with pytest.raises(
        TimeoutError, match="Timeout reached while trying to check for connection .*"
    ):
        wait_for_connection_to_be_available(
            ip_address="17.57.244.197",
            hostname="user",
            path_to_private_key=Path(__file__),
            timeout=1,
            wait_time=1,
            max_retries=1,
            wait_bar=False,
        )


@pytest.mark.filterwarnings("ignore::ResourceWarning")  # Due to boto3
def test_instance_management():
    """Test instance creation."""
    pytest.skip(reason="Issues with AWS")

    with AWSInstance(
        instance_type="t3.nano",
        region_name="eu-west-3",
        instance_name=f"cml_test_aws_deploy-{uuid.uuid4()}",
        verbose=True,
    ) as metadata:
        time.sleep(1)
        assert "ip_address" in metadata


@pytest.mark.filterwarnings("ignore::ResourceWarning")  # Due to boto3
def test_deploy(load_data, tmp_path):  # pylint: disable=too-many-locals,too-many-statements
    """Tests the encrypt decrypt api.

       Arguments:
           load_data (Callable): load data
           tmp_path (Path): temp path
    (Callable): load data
    """
    pytest.skip(reason="Issues with AWS")

    # Easier than taking the list of fitted models
    model_class = LogisticRegression
    n_bits = 2

    # Generate random data
    x, y = load_data(model_class)

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]
    y_test = x[-1:]

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Fit the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_train, y_train)

    # Compile
    extra_params = {"global_p_error": 1 / 100_000}

    fhe_circuit = model.compile(
        x_train,
        **extra_params,
    )
    max_bit_width = fhe_circuit.graph.maximum_integer_bit_width()
    print(f"Max width {max_bit_width}")

    # Instantiate the dev client and server FHEModel client server API
    path_to_model = tmp_path / "dev"
    fhemodel_dev = FHEModelDev(path_dir=path_to_model, model=model)
    fhemodel_dev.save()

    with AWSInstance(
        instance_type="c5.large",
        instance_name=f"cml_test_aws_deploy-{uuid.uuid4()}",
        open_port=5000,
        region_name="eu-west-3",
        verbose=True,
        terminate_on_shutdown=True,
        ami_id=AMI_ID_WITH_CREDENTIALS,
    ) as instance_metadata:

        metadata = deploy_to_aws(
            instance_metadata=instance_metadata,
            path_to_model=path_to_model,
            number_of_ssh_retries=200,
            verbose=True,
            wait_bar=True,
        )

        url = metadata["URL"]
        # Get the necessary data for the client
        # client.zip
        zip_response = requests.get(f"{url}/get_client")
        assert zip_response.status_code == STATUS_OK
        with open(tmp_path / "client.zip", "wb") as file:
            file.write(zip_response.content)

        # Get the data to infer
        assert isinstance(x_test, numpy.ndarray)
        assert isinstance(y_test, numpy.ndarray)

        # Let's create the client
        client = FHEModelClient(path_dir=tmp_path, key_dir=tmp_path / "keys")

        # The client first need to create the private and evaluation keys.
        client.generate_private_and_evaluation_keys()

        # Get the serialized evaluation keys
        serialized_evaluation_keys = client.get_serialized_evaluation_keys()
        assert isinstance(serialized_evaluation_keys, bytes)

        # Evaluation keys can be quite large files but only have to be shared once with the server.

        # Check the size of the evaluation keys (in MB)
        print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")

        # Let's send this evaluation key to the server (this has to be done only once)
        # send_evaluation_key_to_server(serialized_evaluation_keys)

        # Now we have everything for the client to interact with the server
        encrypted_input = None
        clear_input = None

        response = requests.post(
            f"{url}/add_key", files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)}
        )
        assert response.status_code == STATUS_OK
        uid = response.json()["uid"]

        inferences = []
        # Launch the queries
        for i in range(len(x_test)):
            clear_input = x_test[[i], :]

            assert isinstance(clear_input, numpy.ndarray)
            encrypted_input = client.quantize_encrypt_serialize(clear_input)
            assert isinstance(encrypted_input, bytes)

            inferences.append(
                requests.post(
                    f"{url}/compute",
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
        for result in inferences:
            assert result.status_code == STATUS_OK
            encrypted_result = result.content
            decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
            decrypted_predictions.append(decrypted_prediction)

        assert len(decrypted_predictions) == len(x_test)
