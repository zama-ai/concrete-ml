"""Tests the deployment APIs."""

import json
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path
from shutil import copyfile

import numpy
import pytest
import torch
from torch import nn

from concrete import fhe
from concrete.ml.common.utils import CiphertextFormat, to_tuple
from concrete.ml.deployment.fhe_client_server import (
    DeploymentMode,
    FHEModelClient,
    FHEModelDev,
    FHEModelServer,
)
from concrete.ml.pytest.torch_models import EmbeddingModel, FCSmall
from concrete.ml.pytest.utils import (
    MODELS_AND_DATASETS,
    _get_sklearn_tree_models,
    get_model_name,
    instantiate_model_generic,
    is_classifier_or_partial_classifier,
    is_model_class_in_a_list,
)
from concrete.ml.quantization.quantized_module import QuantizedModule
from concrete.ml.sklearn.linear_model import SGDClassifier
from concrete.ml.torch.compile import compile_torch_model

# pylint: disable=too-many-statements,too-many-locals


# Add encrypted training with SGDClassifier manually
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4460
MODELS_AND_DATASETS = MODELS_AND_DATASETS + [
    pytest.param(
        partial(SGDClassifier, fit_encrypted=True, parameters_range=(-1, 1)),
        {"n_samples": 100, "n_features": 10, "n_classes": 2, "n_informative": 10, "n_redundant": 0},
        id="SGDClassifier_Encrypted_Training",
    )
]


class OnDiskNetwork:
    """A network interaction on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.client_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dev_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with

    def dev_send_model_to_server(self):
        """Send the model to the server."""
        copyfile(self.dev_dir.name + "/server.zip", self.server_dir.name + "/server.zip")

    def dev_send_clientspecs_and_modelspecs_to_client(self):
        """Send the clientspecs and evaluation key to the client."""
        copyfile(self.dev_dir.name + "/client.zip", self.client_dir.name + "/client.zip")


# This is a known flaky test
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4014
# This test is disabled on CPU because it is getting stuck
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4737
# @pytest.mark.use_gpu
@pytest.mark.flaky
@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
@pytest.mark.parametrize("n_bits", [2])
def test_client_server_sklearn_inference(
    model_class,
    parameters,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
    check_array_equal,
    check_float_array_equal,
    get_device,
):
    """Test the client-server interface for built-in models' inference."""

    if get_model_name(model_class) == "KNeighborsClassifier":
        # Skipping KNN for this test
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4014
        pytest.skip("Skipping KNN, because FHE predictions and clear ones are differents.")

    # Generate random data
    x, y = load_data(model_class, **parameters)

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    # Instantiate the model
    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Fit the model
    if getattr(model, "fit_encrypted", False):
        model.fit(x_train, y_train, fhe="disable")
    else:
        model.fit(x_train, y_train)

    key_dir = default_configuration.insecure_key_cache_location

    # Running the simulation using a model that is not compiled should not be possible
    with pytest.raises(AttributeError, match=".* model is not compiled.*"):
        check_client_server_inference(
            x_test,
            model,
            key_dir,
            check_array_equal,
            check_float_array_equal,
        )

    compilation_kwargs = {
        "X": x_train,
        "configuration": default_configuration,
        "device": get_device,
    }

    # Compile the model
    fhe_circuit = model.compile(**compilation_kwargs)

    check_input_compression(model, fhe_circuit, is_torch=False, **compilation_kwargs)

    # Check that client and server files are properly generated
    check_client_server_files(model)

    max_bit_width = fhe_circuit.graph.maximum_integer_bit_width()
    print(f"Max width {max_bit_width}")

    # Check that key compression is enabled
    assert os.environ.get("USE_KEY_COMPRESSION") == "1", "'USE_KEY_COMPRESSION' is not enabled"

    # Check with key compression
    check_is_good_execution_for_cml_vs_circuit(x_test, model, simulate=False, n_allowed_runs=1)

    # Check without key compression
    with pytest.MonkeyPatch.context() as mp_context:

        # Disable input ciphertext compression
        mp_context.setenv("USE_KEY_COMPRESSION", "0")

        # Check that input ciphertext compression is disabled
        assert os.environ.get("USE_KEY_COMPRESSION") == "0", "'USE_KEY_COMPRESSION' is not disabled"

        # Compare the FHE predictions with the clear ones. Simulated predictions are not
        # considered in this test.
        check_is_good_execution_for_cml_vs_circuit(x_test, model, simulate=False, n_allowed_runs=1)

    # Check client/server FHE predictions vs the FHE predictions of the dev model
    check_client_server_inference(
        x_test,
        model,
        key_dir,
        check_array_equal,
        check_float_array_equal,
    )


# This is a known flaky test
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4014
@pytest.mark.flaky
@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
@pytest.mark.parametrize("n_bits", [8])
def test_client_server_tfhers_sklearn_inference(
    model_class,
    parameters,
    n_bits,
    load_data,
    default_configuration,
    check_array_equal,
    check_float_array_equal,
):
    """Test the client-server interface.

    This test checks that built-in models inference produces correct results
    when using TFHE-rs ciphertexts for input /output."""

    # Generate random data
    x, y = load_data(model_class, **parameters)

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    # Instantiate the model
    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Fit the model
    if getattr(model, "fit_encrypted", False):
        model.fit(x_train, y_train, fhe="disable")
    else:
        model.fit(x_train, y_train)

    compilation_kwargs = {
        "X": x_train,
        "configuration": default_configuration,
        "ciphertext_format": CiphertextFormat.TFHE_RS,
    }

    if not (
        is_model_class_in_a_list(model_class, _get_sklearn_tree_models())
        and is_classifier_or_partial_classifier(model_class)
    ):
        with pytest.raises(AssertionError, match=".* is only supported for 8-bit tree-based.*"):
            model.compile(**compilation_kwargs)
        return

    # Compile the model
    model.compile(**compilation_kwargs)

    key_dir = default_configuration.insecure_key_cache_location

    # Check client/server FHE predictions vs the FHE predictions of the dev model
    check_client_server_inference(
        x_test,
        model,
        key_dir,
        check_array_equal,
        check_float_array_equal,
        ciphertext_format=CiphertextFormat.TFHE_RS,
    )


def test_client_server_custom_model(
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
    check_array_equal,
    check_float_array_equal,
):
    """Test the client-server interface for a custom model (through a quantized module)."""

    # Generate random data
    x_train, x_test = numpy.random.rand(100, 2), numpy.random.rand(1, 2)

    key_dir = default_configuration.insecure_key_cache_location

    # Running the simulation using a QuantizedModule that is not compiled should not be possible
    with pytest.raises(AttributeError, match=".* quantized module is not compiled.*"):
        # Instantiate an empty QuantizedModule object
        quantized_module = QuantizedModule()

        check_client_server_inference(
            x_test,
            quantized_module,
            key_dir,
            check_array_equal,
            check_float_array_equal,
        )

    torch_model = FCSmall(2, nn.ReLU)

    compilation_kwargs = {
        "torch_inputset": x_train,
        "configuration": default_configuration,
        "n_bits": 2,
    }

    # Get the quantized module from the model and compile it
    quantized_numpy_module = compile_torch_model(torch_model, **compilation_kwargs)

    check_input_compression(
        torch_model, quantized_numpy_module.fhe_circuit, is_torch=True, **compilation_kwargs
    )

    # Check that client and server files are properly generated
    check_client_server_files(quantized_numpy_module)

    # Check that the FHE execution is correct.
    check_is_good_execution_for_cml_vs_circuit(
        x_test, quantized_numpy_module, simulate=False, n_allowed_runs=1
    )

    check_client_server_inference(
        x_test,
        quantized_numpy_module,
        key_dir,
        check_array_equal,
        check_float_array_equal,
    )


def test_client_server_torch_embedding_model(default_configuration):
    """Ensure client/server flow works for a torch model containing an embedding."""

    torch.manual_seed(0)
    num_embeddings = 6
    embedding_dim = 3
    seq_len = 2

    torch_model = EmbeddingModel(num_embeddings, embedding_dim)
    torch_model.eval()

    torch_inputset = torch.randint(0, num_embeddings, size=(8, seq_len)).long()
    sample = torch_inputset[:1].numpy()

    quantized_module = compile_torch_model(
        torch_model,
        torch_inputset,
        configuration=default_configuration,
        n_bits=2,
        rounding_threshold_bits=2,
    )

    network = OnDiskNetwork()
    fhe_model_dev = FHEModelDev(path_dir=network.dev_dir.name, model=quantized_module)
    fhe_model_dev.save()
    network.dev_send_clientspecs_and_modelspecs_to_client()
    network.dev_send_model_to_server()

    key_dir = default_configuration.insecure_key_cache_location
    fhe_model_client = FHEModelClient(path_dir=network.client_dir.name, key_dir=key_dir)
    fhe_model_client.generate_private_and_evaluation_keys(force=True)
    evaluation_keys = fhe_model_client.get_serialized_evaluation_keys(include_tfhers_key=False)
    fhe_model_server = FHEModelServer(path_dir=network.server_dir.name)

    q_x_encrypted_serialized = fhe_model_client.quantize_encrypt_serialize(sample)
    q_y_encrypted_serialized = fhe_model_server.run(q_x_encrypted_serialized, evaluation_keys)

    client_result = fhe_model_client.deserialize_decrypt_dequantize(
        *to_tuple(q_y_encrypted_serialized)
    )
    simulate_result = quantized_module.forward(sample, fhe="simulate")

    numpy.testing.assert_allclose(client_result, simulate_result, atol=1e-2)


def check_client_server_files(model, mode="inference"):
    """Test the client server interface API generates the expected file.

    This test expects that the given model has been trained and compiled in development.
    """
    # Create a new network
    disk_network = OnDiskNetwork()

    # And try to save it again
    fhe_model_dev = FHEModelDev(path_dir=disk_network.dev_dir.name, model=model)
    fhe_model_dev.save(mode=mode, via_mlir=True)

    # Check that re-saving the dev model fails
    with pytest.raises(
        Exception,
        match=(
            f"path_dir: {disk_network.dev_dir.name} is not empty. "
            "Please delete it before saving a new model."
        ),
    ):
        fhe_model_dev.save(mode=mode)

    client_zip_path = Path(disk_network.dev_dir.name) / "client.zip"
    server_zip_path = Path(disk_network.dev_dir.name) / "server.zip"

    # Check that client and server zip files has been generated
    assert (
        client_zip_path.is_file()
    ), f"Client files were not properly generated. Expected {client_zip_path} to be a file."
    assert (
        server_zip_path.is_file()
    ), f"Server files were not properly generated. Expected {server_zip_path} to be a file."

    processing_file_name = "serialized_processing.json"
    versions_file_name = "versions.json"

    # Check that the client.zip file has the processing and versions json files
    with zipfile.ZipFile(client_zip_path) as client_zip:
        with client_zip.open(processing_file_name, "r") as file:
            assert isinstance(
                json.load(file), dict
            ), f"{client_zip_path} does not contain a '{processing_file_name}' file."

        with client_zip.open(versions_file_name, "r") as file:
            assert isinstance(
                json.load(file), dict
            ), f"{client_zip_path} does not contain a '{versions_file_name}' file."

    # Check that the server.zip file has the versions json file
    with zipfile.ZipFile(server_zip_path) as server_zip:
        with server_zip.open("versions.json", "r") as file:
            assert isinstance(
                json.load(file), dict
            ), f"{server_zip_path} does not contain a '{versions_file_name}' file."


def check_client_server_inference(
    x_test,
    model,
    key_dir,
    check_array_equal,
    check_float_array_equal,
    ciphertext_format=CiphertextFormat.CONCRETE,
):
    """Test the client server interface API.

    This test expects that the given model has been trained and compiled in development. It
    basically replicates a production-like interaction and checks that results are on matching the
    development model.

    Args:
        x_test: Input test data.
        model: Trained and compiled model.
        key_dir: Path to key directory.
        check_array_equal: Function to compare arrays with integer values.
        check_float_array_equal: Function to compare arrays with float values.
        ciphertext_format: Specifies the ciphertext backend format. Default to
        CiphertextFormat.CONCRETE.
    """

    # Create a new network
    disk_network = OnDiskNetwork()

    # Save development files
    fhe_model_dev = FHEModelDev(path_dir=disk_network.dev_dir.name, model=model)
    fhe_model_dev.save(mode="inference")

    # Send necessary files to server and client
    disk_network.dev_send_clientspecs_and_modelspecs_to_client()
    disk_network.dev_send_model_to_server()

    # Load the client
    fhe_model_client = FHEModelClient(
        path_dir=disk_network.client_dir.name,
        key_dir=key_dir,
    )
    fhe_model_client.load()

    # Load the server
    fhe_model_server = FHEModelServer(path_dir=disk_network.server_dir.name)
    fhe_model_server.load()

    # Client side : Generate all keys and serialize the evaluation keys for the server
    if ciphertext_format == CiphertextFormat.TFHE_RS:

        evaluation_keys, tfhers_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys(
            include_tfhers_key=True
        )

        assert tfhers_evaluation_keys is not None

        tfhers_sk = fhe_model_client.get_tfhers_secret_key()

        assert tfhers_sk is not None
    else:
        evaluation_keys = fhe_model_client.get_serialized_evaluation_keys(include_tfhers_key=False)

    # Client side : Quantize, encrypt and serialize the data
    q_x_encrypted_serialized = fhe_model_client.quantize_encrypt_serialize(x_test)

    # Server side: Run the model over encrypted data
    q_y_pred_encrypted_serialized = fhe_model_server.run(q_x_encrypted_serialized, evaluation_keys)

    # Client side : Decrypt, de-quantize and post-process the result
    q_y_pred = fhe_model_client.deserialize_decrypt(*to_tuple(q_y_pred_encrypted_serialized))
    y_pred = fhe_model_client.deserialize_decrypt_dequantize(
        *to_tuple(q_y_pred_encrypted_serialized)
    )

    q_x_test = model.quantize_input(x_test)

    # If the model class doesn't have _encrypt_run_decrypt_internal it's a QuantizedModule
    # and doesn't support tfhe-rs interop
    if getattr(model, "_encrypt_run_decrypt_internal", False):
        q_y_pred_dev = model._encrypt_run_decrypt_internal(  # pylint: disable=protected-access
            q_x_test
        )
    else:
        assert isinstance(model, QuantizedModule)
        q_y_pred_dev = model.fhe_circuit.encrypt_run_decrypt(q_x_test)

    y_pred_dev = model.dequantize_output(q_y_pred_dev)
    y_pred_dev = model.post_processing(y_pred_dev)

    check_array_equal(q_y_pred, q_y_pred_dev)

    # Check that both quantized and de-quantized (+ post-processed) results from the server are
    # matching the ones from the dec model
    check_float_array_equal(y_pred, y_pred_dev)


def check_input_compression(model, fhe_circuit_compressed, is_torch, **compilation_kwargs):
    """Check that input compression properly reduces input sizes."""

    # Check that input ciphertext compression is enabled
    assert os.environ.get("USE_INPUT_COMPRESSION") == "1", "'USE_INPUT_COMPRESSION' is not enabled"

    compressed_size = fhe_circuit_compressed.size_of_inputs

    with pytest.MonkeyPatch.context() as mp_context:

        # Disable input ciphertext compression
        mp_context.setenv("USE_INPUT_COMPRESSION", "0")

        # Check that input ciphertext compression is disabled
        assert (
            os.environ.get("USE_INPUT_COMPRESSION") == "0"
        ), "'USE_INPUT_COMPRESSION' is not disabled"

        if is_torch:
            fhe_circuit_uncompressed = compile_torch_model(model, **compilation_kwargs).fhe_circuit
        else:
            fhe_circuit_uncompressed = model.compile(**compilation_kwargs)

        uncompressed_size = fhe_circuit_uncompressed.size_of_inputs

    # Make sure inputs are compressed by a given important factor
    input_compression_factor = 50

    assert input_compression_factor * compressed_size < uncompressed_size, (
        "Compressed input ciphertext's is not smaller than the uncompressed input ciphertext. Got "
        f"{compressed_size} bytes (compressed) and {uncompressed_size} bytes (uncompressed)."
    )


ERROR_MSG_BAD_MODE = "Mode must be either 'inference' or 'training'"
ERROR_MSG_NO_FHE_CIRCUIT = "Training FHE circuit does not exist."


@pytest.mark.parametrize("n_bits", [2])
@pytest.mark.parametrize(
    "mode, fit_encrypted, error_message",
    [
        ("invalid_mode", True, ERROR_MSG_BAD_MODE),
        ("INVALID_MODE", True, ERROR_MSG_BAD_MODE),
        ("train", True, ERROR_MSG_BAD_MODE),
        ("", True, ERROR_MSG_BAD_MODE),
        (None, True, None),
        ("inference", False, None),
        ("inference", True, None),
        ("training", False, ERROR_MSG_NO_FHE_CIRCUIT),
        ("training", True, None),
        (DeploymentMode.INFERENCE, False, None),
        (DeploymentMode.TRAINING, False, ERROR_MSG_NO_FHE_CIRCUIT),
        (DeploymentMode.TRAINING, True, None),
    ],
)
def test_save_mode_handling(n_bits, fit_encrypted, mode, error_message):
    """Test that the save method handles valid and invalid modes correctly."""

    # Generate random data
    x, y = numpy.random.rand(20, 2), numpy.random.randint(0, 2, 20)

    x_train = x[:-1]
    y_train = y[:-1]

    # Instantiate the model
    parameters_range = [-1, 1] if fit_encrypted else None
    model = instantiate_model_generic(
        partial(SGDClassifier, fit_encrypted=fit_encrypted, parameters_range=parameters_range),
        n_bits=n_bits,
    )

    # Fit the model in the clear
    if getattr(model, "fit_encrypted", False):
        model.fit(x_train, y_train, fhe="disable")
    else:
        model.fit(x_train, y_train)

    # Compile
    model.compile(X=x_train)

    # Create FHEModelDev instance
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dev = FHEModelDev(path_dir=temp_dir, model=model)

        if error_message:
            with pytest.raises((AssertionError, ValueError), match=error_message):
                model_dev.save(mode=mode)
        else:
            model_dev.save(mode=mode)


def quantize_encrypt_training_inputs(
    x,
    y,
    weights,
    bias,
    batch_size=8,
    max_iter=None,
    fhe_client=None,
    quantized_module=None,
):
    """Quantize and encrypt training data, and serialize them if in client-server mode."""

    assert (fhe_client is None) ^ (
        quantized_module is None
    ), "Either provide a client or a QuantizedModule instance"

    x_batches_enc, y_batches_enc = [], []

    for i in range(0, x.shape[0], batch_size):

        # Avoid the last batch if it's not a multiple of 'batch_size'
        if i + batch_size < x.shape[0]:
            batch_range = range(i, i + batch_size)
        else:
            break

        # Make the data X (1, batch_size, n_features) and y (1, batch_size, n_targets=1)
        x_batch = numpy.expand_dims(x[batch_range, :], 0)
        y_batch = numpy.expand_dims(y[batch_range], (0, 2))

        # Quantize and encrypt the batch
        # Serialize as well if in client-server mode
        if fhe_client is not None:
            x_batch_enc, y_batch_enc, _, _ = fhe_client.quantize_encrypt_serialize(
                x_batch, y_batch, None, None
            )
        else:
            q_x_batch, q_y_batch, _, _ = quantized_module.quantize_input(
                x_batch, y_batch, None, None
            )
            x_batch_enc, y_batch_enc, _, _ = quantized_module.fhe_circuit.encrypt(
                q_x_batch, q_y_batch, None, None
            )

        x_batches_enc.append(x_batch_enc)
        y_batches_enc.append(y_batch_enc)

        # Stop at 'max_iter' iterations
        if max_iter is not None and (i // batch_size) >= max_iter - 1:
            break

    # Quantize and encrypt the weight and bias values
    # Serialize as well if in client-server mode
    if fhe_client is not None:
        _, _, weights_enc, bias_enc = fhe_client.quantize_encrypt_serialize(
            None, None, weights, bias
        )
    else:
        _, _, q_weights, q_bias = quantized_module.quantize_input(
            None,
            None,
            weights,
            bias,
        )
        _, _, weights_enc, bias_enc = quantized_module.fhe_circuit.encrypt(
            None, None, q_weights, q_bias
        )

    return x_batches_enc, y_batches_enc, weights_enc, bias_enc


def fhe_training_run(
    x_batches_enc,
    y_batches_enc,
    weights_enc,
    bias_enc,
    evaluation_keys=None,
    fhe_server=None,
    quantized_module=None,
):
    """Run encrypted training for several iterations."""

    assert (fhe_server is None) ^ (
        quantized_module is None
    ), "Either provide a server or a QuantizedModule instance"

    # Deserialize weights, bias and evaluations keys if in client-server mode
    if fhe_server is not None:
        weights_enc = fhe.Value.deserialize(weights_enc)
        bias_enc = fhe.Value.deserialize(bias_enc)

        assert evaluation_keys is not None, "Please provide evaluations keys in client-server mode"

        evaluation_keys = fhe.EvaluationKeys.deserialize(evaluation_keys)

    # Run the circuit on the server n times, n being the number of batches provided
    for x_batch, y_batch in zip(x_batches_enc, y_batches_enc):

        # Deserialize the input batches if in client-server mode
        if fhe_server is not None:
            x_batch = fhe.Value.deserialize(x_batch)
            y_batch = fhe.Value.deserialize(y_batch)

            weights_enc, bias_enc = fhe_server.run(
                (x_batch, y_batch, weights_enc, bias_enc), evaluation_keys
            )
        else:
            weights_enc, bias_enc = quantized_module.fhe_circuit.run(
                x_batch, y_batch, weights_enc, bias_enc
            )

    # Serialize the output weight and bias values if in client-server mode
    if fhe_server is not None:
        weights_enc = weights_enc.serialize()
        bias_enc = bias_enc.serialize()

    return weights_enc, bias_enc


def decrypt_dequantize_training_outputs(
    weights_enc,
    bias_enc,
    fhe_client=None,
    quantized_module=None,
):
    """Decrypt and de-quantize training outputs, and de-serialize them if in client-server mode."""
    if fhe_client is not None:
        q_weights, q_bias = fhe_client.deserialize_decrypt(weights_enc, bias_enc)
        weights, bias = fhe_client.deserialize_decrypt_dequantize(weights_enc, bias_enc)
    else:
        q_weights, q_bias = quantized_module.fhe_circuit.decrypt(weights_enc, bias_enc)
        weights, bias = quantized_module.dequantize_output(q_weights, q_bias)

    return q_weights, q_bias, weights, bias


def get_fitted_weights(
    x_train,
    y_train,
    weights,
    bias,
    batch_size=None,
    max_iter=None,
    fhe_client=None,
    fhe_server=None,
    quantized_module=None,
):
    """RunFHE training in client-server or un development mode."""

    # Client side : Quantize, encrypt and serialize the data
    x_batches_enc, y_batches_enc, weights_enc, bias_enc = quantize_encrypt_training_inputs(
        x_train,
        y_train,
        weights,
        bias,
        batch_size=batch_size,
        max_iter=max_iter,
        fhe_client=fhe_client,
        quantized_module=quantized_module,
    )

    evaluation_keys = None

    # Client side : Generate all keys and serialize the evaluation keys for the server
    if fhe_client is not None:
        evaluation_keys = fhe_client.get_serialized_evaluation_keys()

    # Server side: Fit the model over encrypted data using the training FHE circuit
    weights_enc, bias_enc = fhe_training_run(
        x_batches_enc,
        y_batches_enc,
        weights_enc,
        bias_enc,
        evaluation_keys=evaluation_keys,
        fhe_server=fhe_server,
        quantized_module=quantized_module,
    )

    # Client side: Deserialize, decrypt and de-quantize the result
    q_weights, q_bias, weights, bias = decrypt_dequantize_training_outputs(
        weights_enc,
        bias_enc,
        fhe_client=fhe_client,
        quantized_module=quantized_module,
    )

    return q_weights, q_bias, weights, bias


def check_client_server_training(
    model,
    x_train,
    y_train,
    weights,
    bias,
    batch_size,
    max_iter,
    key_dir,
    check_array_equal,
    check_float_array_equal,
):
    """Test the client server interface API for encrypted training."""

    model_name = get_model_name(model)
    assert hasattr(
        model, "training_quantized_module"
    ), f"Model '{model_name}' has no 'training_quantized_module' attribute"

    assert (
        model.training_quantized_module is not None
    ), f"Attribute 'training_quantized_module' for model '{model_name}' has not been set"

    # Create a new network
    disk_network = OnDiskNetwork()

    # Save development files
    fhe_dev = FHEModelDev(path_dir=disk_network.dev_dir.name, model=model)
    fhe_dev.save(mode="training", via_mlir=True)

    # Send necessary files to server and client
    disk_network.dev_send_clientspecs_and_modelspecs_to_client()
    disk_network.dev_send_model_to_server()

    # Load the client
    fhe_client = FHEModelClient(
        path_dir=disk_network.client_dir.name,
        key_dir=key_dir,
    )
    fhe_client.load()

    # Load the server
    fhe_server = FHEModelServer(path_dir=disk_network.server_dir.name)
    fhe_server.load()

    # Client-server training
    (
        q_weights_deployment,
        q_bias_deployment,
        weights_deployment,
        bias_deployment,
    ) = get_fitted_weights(
        x_train,
        y_train,
        weights,
        bias,
        batch_size=batch_size,
        max_iter=max_iter,
        fhe_client=fhe_client,
        fhe_server=fhe_server,
    )

    # Quantized module (development) training
    (
        q_weights_development,
        q_bias_development,
        weights_development,
        bias_development,
    ) = get_fitted_weights(
        x_train,
        y_train,
        weights,
        bias,
        batch_size=batch_size,
        max_iter=max_iter,
        quantized_module=model.training_quantized_module,
    )

    # Check that both quantized outputs from the quantized module (development) are matching the
    # ones from the deployment interface
    check_array_equal(q_weights_deployment, q_weights_development)
    check_array_equal(q_bias_deployment, q_bias_development)

    # Same for de-quantized outputs
    check_float_array_equal(weights_deployment, weights_development)
    check_float_array_equal(bias_deployment, bias_development)


@pytest.mark.parametrize(
    "model_class, parameters",
    [
        pytest.param(
            partial(SGDClassifier, fit_encrypted=True, parameters_range=(-1, 1)),
            {
                "n_samples": 100,
                "n_features": 2,
                "n_classes": 2,
                "n_informative": 2,
                "n_redundant": 0,
            },
            id="SGDClassifier_Encrypted_Training",
        )
    ],
)
@pytest.mark.parametrize("n_bits", [2])
def test_client_server_sklearn_training(
    model_class,
    parameters,
    n_bits,
    load_data,
    default_configuration,
    check_array_equal,
    check_float_array_equal,
):
    """Test the client-server interface for encrypted training."""
    max_iter = 2
    batch_size = 2

    # Generate random data
    x_train, y_train = load_data(model_class, **parameters)

    # Instantiate the model
    model = instantiate_model_generic(model_class, n_bits=n_bits, max_iter=max_iter)

    # Set a higher p_error s that tests pass
    model.training_p_error = 2 ** (-40)

    # SGDClassifier cannot set teh training batch size and number of bits through the initializer,
    # so we fix a lower value in order to speed-up tests, especially since we do not actually check
    # any score here
    model.batch_size = batch_size
    model.n_bits_training = n_bits

    # Generate the min and max values for x_train and y_train
    x_min, x_max = x_train.min(axis=0), x_train.max(axis=0)
    y_min, y_max = y_train.min(), y_train.max()

    # Create a dataset with the min and max values for each feature, repeated to fill the batch size
    x_compile_set = numpy.vstack([x_min, x_max] * (batch_size // 2))

    # Create a dataset with the min and max values for y, repeated to fill the batch size
    y_compile_set = numpy.array([y_min, y_max] * (batch_size // 2))

    # Fit the model with the created dataset to compile it for production
    # This step ensures the model knows the number of features, targets and features distribution
    # Remove this once this step is improved
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4466
    model.fit(x_compile_set, y_compile_set, fhe="disable")

    # Check that client and server files are properly generated
    check_client_server_files(model, mode="training")

    # Initialize the weight and bias randomly
    # They are going to be updated using FHE training.
    weights = numpy.random.rand(1, x_train.shape[1], 1)
    bias = numpy.random.rand(1, 1, 1)

    key_dir = default_configuration.insecure_key_cache_location

    # Check client/server FHE training
    check_client_server_training(
        model,
        x_train,
        y_train,
        weights,
        bias,
        batch_size,
        max_iter,
        key_dir,
        check_array_equal,
        check_float_array_equal,
    )
