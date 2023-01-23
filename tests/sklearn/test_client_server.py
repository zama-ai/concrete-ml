"""Tests the deployment APIs."""

import tempfile
import warnings
from functools import partial
from shutil import copyfile

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning
from torch import nn

from concrete.ml.common.utils import get_model_name, is_model_class_in_a_list
from concrete.ml.deployment.fhe_client_server import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.pytest.torch_models import FCSmall
from concrete.ml.pytest.utils import sklearn_models_and_datasets
from concrete.ml.sklearn.base import get_sklearn_neural_net_models
from concrete.ml.torch.compile import compile_torch_model

# pylint: disable=too-many-statements


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
        copyfile(
            self.dev_dir.name + "/serialized_processing.json",
            self.client_dir.name + "/serialized_processing.json",
        )

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()


@pytest.mark.parametrize("model, parameters", sklearn_models_and_datasets)
def test_client_server_sklearn(
    default_configuration_no_jit,
    model,
    parameters,
    load_data,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Tests the encrypt decrypt api."""

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2756
    # We currently have issues with some TweedieRegressor's
    if isinstance(model, partial):
        if model.func.__name__ == "TweedieRegressor":
            if model.keywords.get("link") == "log" or model.keywords.get("power") != 0:
                pytest.skip("Waiting for #2756 fix")

    # Generate random data
    model_class = model
    model_name = get_model_name(model_class)
    x, y = load_data(**parameters, model_name=model_name)

    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        x = x.astype(numpy.float32)

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    model = model_class()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model.fit(x_train, y_train)

    # Compile
    extra_params = {"global_p_error": 1 / 100_000}

    fhe_circuit = model.compile(
        x_train, default_configuration_no_jit, **extra_params, show_mlir=True
    )
    max_bit_width = fhe_circuit.graph.maximum_integer_bit_width()
    print(f"Max width {max_bit_width}")

    # Check that the FHE execution is correct.
    # With a global_p_error of 1/100_000 we only allow one run.
    check_is_good_execution_for_cml_vs_circuit(x_test, model, n_allowed_runs=1)
    client_server_simulation(x_train, x_test, model, default_configuration_no_jit)


def test_client_server_custom_model(
    default_configuration_no_jit, check_is_good_execution_for_cml_vs_circuit
):
    """Tests the client server custom model."""

    # Generate random data
    x_train, x_test = numpy.random.rand(100, 2), numpy.random.rand(1, 2)
    torch_model = FCSmall(2, nn.ReLU)
    n_bits = 2

    # Get the quantized module from the model
    quantized_numpy_module = compile_torch_model(
        torch_model,
        x_train,
        configuration=default_configuration_no_jit,
        n_bits=n_bits,
        use_virtual_lib=False,
        global_p_error=1 / 100_000,
    )

    # Check that the FHE execution is correct.
    # With a global_p_error of 1/100_000 we only allow one run.
    q_x_test = quantized_numpy_module.quantize_input(x_test)
    check_is_good_execution_for_cml_vs_circuit(q_x_test, quantized_numpy_module, n_allowed_runs=1)
    client_server_simulation(x_train, x_test, quantized_numpy_module, default_configuration_no_jit)


def client_server_simulation(x_train, x_test, model, default_configuration_no_jit):
    """Simulate the client server interaction."""
    # Model has been trained and compiled on the server.
    # Now we use the fhe api to go into production.

    # Set up the fake network
    network = OnDiskNetwork()

    # Instantiate the dev client and server FHEModel client server API
    fhemodel_dev = FHEModelDev(path_dir=network.dev_dir.name, model=model)
    fhemodel_dev.save()

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
        key_dir=default_configuration_no_jit.insecure_key_cache_location,
    )
    fhemodel_client.load()

    # Grab the model and save it again (feature to allow compilation on an adventurous OS)
    client_model = fhemodel_client.model
    client_model.fhe_circuit = model.fhe_circuit

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
        key_dir=default_configuration_no_jit.insecure_key_cache_location,
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

    # Decrypt and dequantize the result
    y_pred_on_client_quantized = fhemodel_client.deserialize_decrypt(serialized_result)
    y_pred_on_client_dequantized = fhemodel_client.deserialize_decrypt_dequantize(serialized_result)

    # Get the y_pred_model_server_clear

    # Predict based on the model we are testing
    qtest = model.quantize_input(x_test)
    y_pred_model_server_ds_quantized = model.fhe_circuit.encrypt_run_decrypt(qtest)
    y_pred_model_server_ds_dequantized = model.post_processing(y_pred_model_server_ds_quantized)

    # Make sure the quantized predictions are the same for the dev model and server
    numpy.testing.assert_array_equal(y_pred_on_client_quantized, y_pred_model_server_ds_quantized)
    numpy.testing.assert_array_equal(
        y_pred_on_client_dequantized, y_pred_model_server_ds_dequantized
    )

    # Clean up
    network.cleanup()
