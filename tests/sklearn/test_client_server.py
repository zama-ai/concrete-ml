"""Tests the deployment APIs."""

import tempfile
import warnings
from functools import partial
from shutil import copyfile

import numpy
import pytest
from shared import classifiers, regressors
from sklearn.exceptions import ConvergenceWarning
from torch import nn

from concrete.ml.deployment.fhe_client_server import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    RandomForestClassifier,
    XGBClassifier,
)
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


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
@pytest.mark.parametrize("p_error", [6.3342483999973e-05, 1e-04, None])
def test_client_server_sklearn(default_configuration_no_jit, model, parameters, load_data, p_error):
    """Tests the encrypt decrypt api."""

    # Generate random data
    x, y = load_data(**parameters)
    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    if isinstance(model, partial):
        if model.func in [NeuralNetClassifier, NeuralNetRegressor]:
            model_params = model.keywords
            # Change module__input_dim to be the same as the input dimension
            model_params["module__input_dim"] = x_train.shape[1]
            # qnns require float32 as input
            x = x.astype(numpy.float32)
            x_train = x_train.astype(numpy.float32)
            x_test = x_test.astype(numpy.float32)

            if model.func is NeuralNetRegressor:
                # Reshape y_train and y_test if 1d (regression for neural nets)
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1).astype(numpy.float32)
    elif model in [XGBClassifier, RandomForestClassifier]:
        model_params = {
            "n_estimators": 5,
            "max_depth": 2,
            "random_state": numpy.random.randint(0, 2**15),
        }
    elif model is DecisionTreeClassifier:
        model_params = {"max_depth": 2, "random_state": numpy.random.randint(0, 2**15)}
    elif model in [LogisticRegression]:
        model_params = {"random_state": numpy.random.randint(0, 2**15)}
    else:
        model_params = {}

    model = model(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model.fit(x_train, y_train)

    # Compile
    extra_params = {}
    if p_error is not None:
        extra_params["p_error"] = p_error

    fhe_circuit = model.compile(
        x_train, default_configuration_no_jit, **extra_params, show_mlir=True
    )
    max_bit_width = fhe_circuit.graph.maximum_integer_bit_width()
    print(f"Max width {max_bit_width}")

    client_server_simulation(x_train, x_test, model, default_configuration_no_jit)


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self, input_output, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=input_output)
        self.act_f = activation_function()
        self.fc2 = nn.Linear(in_features=input_output, out_features=input_output)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.act_f(out)
        out = self.fc2(out)

        return out


@pytest.mark.parametrize("p_error", [6.3342483999973e-05, 1e-04, None])
def test_client_server_custom_model(default_configuration_no_jit, p_error):
    """Tests the client server custom model."""

    # Generate random data
    x_train, x_test = numpy.random.rand(100, 2), numpy.random.rand(1, 2)
    torch_model = FC(2, nn.ReLU)
    n_bits = 2

    # Get the quantized module from the model
    extra_params = {}
    if p_error is not None:
        extra_params["p_error"] = p_error

    quantized_numpy_module = compile_torch_model(
        torch_model,
        x_train,
        configuration=default_configuration_no_jit,
        n_bits=n_bits,
        use_virtual_lib=False,
        **extra_params,
    )

    client_server_simulation(x_train, x_test, quantized_numpy_module, default_configuration_no_jit)


def client_server_simulation(x_train, x_test, model, default_configuration_no_jit):
    """Simulate the client server interaction."""
    # Model has been trained and compiled on the server.
    # Now we use the fheapi to go into production.

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
    y_pred_on_client = fhemodel_client.deserialize_decrypt_dequantize(serialized_result)

    # Get the y_pred_model_server_clear

    # Predict based on the model we are testing
    if hasattr(model, "_estimator_type"):
        if model._estimator_type == "classifier":  # pylint: disable=protected-access
            y_pred_model_server_ds = model.predict_proba(x_test, execute_in_fhe=True)
        else:
            y_pred_model_server_ds = model.predict(x_test, execute_in_fhe=True)
    else:
        qtest = model.quantize_input(x_test)
        y_pred_model_server_ds = model.forward_fhe.encrypt_run_decrypt(qtest)
        y_pred_model_server_ds = model.dequantize_output(y_pred_model_server_ds)

    # Round to 5th decimal place (torch and numpy mismatch)
    y_pred_model_server_ds = numpy.round(y_pred_model_server_ds, (5))
    y_pred_on_client = numpy.round(y_pred_on_client, (5))

    numpy.testing.assert_array_equal(y_pred_on_client, y_pred_model_server_ds)

    # Clean up
    network.cleanup()
