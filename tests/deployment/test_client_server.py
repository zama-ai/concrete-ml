"""Tests the deployment APIs."""

import json
import tempfile
import warnings
import zipfile
from pathlib import Path
from shutil import copyfile

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning
from torch import nn

from concrete.ml.deployment.fhe_client_server import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.pytest.torch_models import FCSmall
from concrete.ml.pytest.utils import (
    get_model_name,
    instantiate_model_generic,
    sklearn_models_and_datasets,
)
from concrete.ml.quantization.quantized_module import QuantizedModule
from concrete.ml.torch.compile import compile_torch_model

# pylint: disable=too-many-statements,too-many-locals


class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.client_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dev_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def client_send_evaluation_key_to_server(self, serialized_evaluation_keys):
        """Send the public key to the server."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "wb") as f:
            f.write(serialized_evaluation_keys)

    def client_send_input_to_server_for_prediction(self, encrypted_input):
        """Send the input to the server."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "rb") as f:
            serialized_evaluation_keys = f.read()
        encrypted_prediction = FHEModelServer(self.server_dir.name).run(
            encrypted_input, serialized_evaluation_keys
        )
        with open(self.server_dir.name + "/encrypted_prediction.enc", "wb") as f:
            f.write(encrypted_prediction)

    def dev_send_model_to_server(self):
        """Send the model to the server."""
        copyfile(self.dev_dir.name + "/server.zip", self.server_dir.name + "/server.zip")

    def server_send_encrypted_prediction_to_client(self):
        """Send the encrypted prediction to the client."""
        with open(self.server_dir.name + "/encrypted_prediction.enc", "rb") as f:
            encrypted_prediction = f.read()
        return encrypted_prediction

    def dev_send_clientspecs_and_modelspecs_to_client(self):
        """Send the clientspecs and evaluation key to the client."""
        copyfile(self.dev_dir.name + "/client.zip", self.client_dir.name + "/client.zip")

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()


# Remove this flaky flag once the KNN issue is fixed
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4009
@pytest.mark.flaky
@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize("n_bits", [3])
def test_client_server_sklearn(
    default_configuration,
    model_class,
    parameters,
    n_bits,
    load_data,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Tests the encrypt decrypt api."""

    # Generate random data
    x, y = load_data(model_class, **parameters)

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Fit the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_train, y_train)

    # Compile
    extra_params = {"global_p_error": 1 / 100_000}

    # Running the simulation using a model that is not compiled should not be possible
    with pytest.raises(AttributeError, match=".* model is not compiled.*"):
        client_server_simulation(x_train, x_test, model, default_configuration)

    fhe_circuit = model.compile(
        x_train, default_configuration, **extra_params, show_mlir=(n_bits <= 8)
    )

    if get_model_name(model) == "KNeighborsClassifier":
        # Fit the model
        with warnings.catch_warnings():
            # Sometimes, we miss convergence, which is not a problem for our test
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x, y)

    max_bit_width = fhe_circuit.graph.maximum_integer_bit_width()
    print(f"Max width {max_bit_width}")

    # Check that the FHE execution is correct.
    # With a global_p_error of 1/100_000 we only allow one run.
    check_is_good_execution_for_cml_vs_circuit(x_test, model, simulate=False, n_allowed_runs=1)
    client_server_simulation(x_train, x_test, model, default_configuration)


def test_client_server_custom_model(
    default_configuration, check_is_good_execution_for_cml_vs_circuit
):
    """Tests the client server custom model."""

    # Generate random data
    x_train, x_test = numpy.random.rand(100, 2), numpy.random.rand(1, 2)

    # Running the simulation using a QuantizedModule that is not compiled should not be possible
    with pytest.raises(AttributeError, match=".* quantized module is not compiled.*"):
        # Instantiate an empty QuantizedModule object
        quantized_module = QuantizedModule()

        client_server_simulation(x_train, x_test, quantized_module, default_configuration)

    torch_model = FCSmall(2, nn.ReLU)
    n_bits = 2

    # Get the quantized module from the model
    quantized_numpy_module = compile_torch_model(
        torch_model,
        x_train,
        configuration=default_configuration,
        n_bits=n_bits,
        global_p_error=1 / 100_000,
    )

    # Check that the FHE execution is correct.
    # With a global_p_error of 1/100_000 we only allow one run.
    check_is_good_execution_for_cml_vs_circuit(
        x_test, quantized_numpy_module, simulate=False, n_allowed_runs=1
    )

    client_server_simulation(x_train, x_test, quantized_numpy_module, default_configuration)


def client_server_simulation(x_train, x_test, model, default_configuration):
    """Simulate the client server interaction."""
    # Model has been trained and compiled on the server.
    # Now we use the fhe api to go into production.

    # Set up the fake network
    network = OnDiskNetwork()

    # Instantiate the dev client and server FHEModel client server API
    fhemodel_dev = FHEModelDev(path_dir=network.dev_dir.name, model=model)
    fhemodel_dev.save()

    # Check that the processing json file is in the client.zip file
    with zipfile.ZipFile(Path(network.dev_dir.name) / "client.zip") as client_zip:
        with client_zip.open("serialized_processing.json", "r") as file:
            assert isinstance(json.load(file), dict)

    # Send necessary files to server and client
    network.dev_send_clientspecs_and_modelspecs_to_client()
    network.dev_send_model_to_server()

    # Make sure the save fails now that the folder is populated
    err_msg = (
        f"path_dir: {network.dev_dir.name} is not empty."
        "Please delete it before saving a new model."
    )
    with pytest.raises(Exception, match=err_msg):
        fhemodel_dev.save()

    fhemodel_client = FHEModelClient(
        path_dir=network.client_dir.name,
        key_dir=default_configuration.insecure_key_cache_location,
    )
    fhemodel_client.load()

    # Grab the model and save it again
    # No user is expected to load a FHEModelDev instance from a FHEModelClient's model. This is
    # only made a testing for making sure the model has the expected attributes
    client_model = fhemodel_client.model
    client_model.fhe_circuit = model.fhe_circuit

    # pylint: disable-next=protected-access
    client_model._is_compiled = True

    network.cleanup()

    # Create a new network
    network = OnDiskNetwork()

    # And try to save it again
    fhemodel_dev_ = FHEModelDev(path_dir=network.dev_dir.name, model=client_model)
    fhemodel_dev_.save()

    # Send necessary files to server and client
    network.dev_send_clientspecs_and_modelspecs_to_client()
    network.dev_send_model_to_server()

    # And try to load it again
    fhemodel_client_ = FHEModelClient(
        path_dir=network.client_dir.name,
        key_dir=default_configuration.insecure_key_cache_location,
    )
    fhemodel_client_.load()

    # Now we can also load the server part
    fhemodel_server = FHEModelServer(path_dir=network.server_dir.name)
    fhemodel_server.load()

    # Make sure the client has the exact same quantization as the server
    qx_ref_model = model.quantize_input(x_train)
    qx_client = fhemodel_client.model.quantize_input(x_train)
    qx_dev = fhemodel_dev.model.quantize_input(x_train)
    numpy.testing.assert_array_equal(qx_ref_model, qx_client)
    numpy.testing.assert_array_equal(qx_ref_model, qx_dev)

    # Create evaluation keys for the server
    fhemodel_client.generate_private_and_evaluation_keys()

    # Get the server evaluation key
    serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()

    # Encrypt new data
    serialized_qx_new_encrypted = fhemodel_client.quantize_encrypt_serialize(x_test)

    # Here data can be saved, sent over the network, etc.

    # Now back to the server

    # Run the model over encrypted data
    serialized_result = fhemodel_server.run(serialized_qx_new_encrypted, serialized_evaluation_keys)

    # Back to the client

    # Decrypt and de-quantize the result
    y_pred_on_client_quantized = fhemodel_client.deserialize_decrypt(serialized_result)
    y_pred_on_client_dequantized = fhemodel_client.deserialize_decrypt_dequantize(serialized_result)

    # Get the y_pred_model_server_clear

    # Predict based on the model we are testing
    qtest = model.quantize_input(x_test)
    y_pred_model_server_ds_quantized = model.fhe_circuit.encrypt_run_decrypt(qtest)
    y_pred_model_server_ds_dequantized = model.dequantize_output(y_pred_model_server_ds_quantized)
    y_pred_model_server_ds_dequantized = model.post_processing(y_pred_model_server_ds_dequantized)

    # Make sure the quantized predictions are the same for the dev model and server
    numpy.testing.assert_array_equal(y_pred_on_client_quantized, y_pred_model_server_ds_quantized)
    numpy.testing.assert_array_equal(
        y_pred_on_client_dequantized, y_pred_model_server_ds_dequantized
    )

    # Make sure the clear predictions are the same for the server
    if get_model_name(model) == "KNeighborsClassifier":
        y_pred_model_clear = model.predict(x_test, fhe="disable")
        numpy.testing.assert_array_equal(y_pred_model_clear, y_pred_model_server_ds_dequantized)

    # Clean up
    network.cleanup()
