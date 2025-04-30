"""Tests for the hybrid model converter."""

import sys
import tempfile
from pathlib import Path
from typing import List, Union

import numpy
import pytest
import torch
from concrete.fhe import Configuration
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from concrete.ml.pytest.torch_models import FCSmall, PartialQATModel
from concrete.ml.torch.hybrid_model import (
    HybridFHEModel,
    tuple_to_underscore_str,
    underscore_str_to_tuple,
)


@pytest.mark.parametrize(
    "tup",
    [
        tuple(),
        (1,),
        (1, 2),
        (1, (2, 3)),
        ((1, 2), (3, 4, 5)),
    ],
)
def test_tuple_serialization(tup):
    """Test that tuple serialization is correctly handled."""
    assert tup == underscore_str_to_tuple(tuple_to_underscore_str(tup))


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches
def run_hybrid_llm_test(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    module_names: Union[str, List[str]],
    expected_accuracy: float,
    has_pbs: bool,
    transformers_installed: bool,
    glwe_backend_installed: bool,
    monkeypatch: pytest.MonkeyPatch,
    get_device: str,
):
    """Run the test for any model with its private module names."""

    # Configure the model
    configuration = Configuration(
        single_precision=False,
        compress_input_ciphertexts=True,
    )

    # Mock sys.modules to simulate missing modules
    if not transformers_installed:
        monkeypatch.setitem(sys.modules, "transformers", None)
    if not glwe_backend_installed:
        monkeypatch.setitem(sys.modules, "concrete_ml_extensions", None)

    # Initialize and compile the hybrid model
    hybrid_model = HybridFHEModel(model, module_names)

    try:
        hybrid_model.compile_model(
            inputs,
            p_error=10e-40,
            n_bits=9,
            rounding_threshold_bits=8,
            configuration=configuration,
            device=get_device,
        )
    except RuntimeError as error:
        # Skip test if NoParametersFound error occurs
        if "NoParametersFound" in str(error):
            pytest.skip(str(error))
        else:
            raise

    # Run the model in different modes
    logits_simulate = None
    if has_pbs or not glwe_backend_installed:
        logits_simulate = hybrid_model(inputs, fhe="simulate").logits
    else:
        with pytest.raises(AssertionError, match=".*fhe=simulate is not supported.*"):
            hybrid_model(inputs, fhe="simulate")

    logits_disable = hybrid_model(inputs, fhe="disable").logits
    logits_original = hybrid_model(inputs, fhe="torch").logits

    # Check programmable bootstrap counts if not glwe backend
    if not glwe_backend_installed:
        for module in hybrid_model.private_q_modules.values():
            pbs_count = module.fhe_circuit.statistics.get("programmable_bootstrap_count", 0)
            if has_pbs:
                assert pbs_count > 0, "Expected programmable bootstrap count > 0"
            else:
                assert pbs_count == 0, "Expected programmable bootstrap count == 0"

    # Compare top-k accuracy
    k = 5
    topk_disable = logits_disable.topk(k, dim=-1).indices
    topk_original = logits_original.topk(k, dim=-1).indices
    accuracy_disable = (topk_disable == topk_original).float().mean().item()
    assert (
        accuracy_disable >= expected_accuracy
    ), f"Disable accuracy {accuracy_disable:.4f} is below expected {expected_accuracy:.4f}"

    if logits_simulate is not None:
        assert torch.allclose(logits_disable, logits_simulate, atol=1e-7)
        topk_simulate = logits_simulate.topk(k, dim=-1).indices
        accuracy_simulate = (topk_simulate == topk_original).float().mean().item()
        assert (
            accuracy_simulate >= expected_accuracy
        ), f"Simulate accuracy {accuracy_simulate:.4f} is below expected {expected_accuracy:.4f}"

    # Test model saving and deployment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        if not has_pbs and glwe_backend_installed:
            hybrid_model.save_and_clear_private_info(temp_dir_path)
        else:
            # If transformers is not installed, skip the saving test
            if not transformers_installed:
                pytest.skip("Skipping save test as transformers module is not available")

            hybrid_model.save_and_clear_private_info(temp_dir_path)
            hybrid_model.set_fhe_mode("remote")

            # Verify saved files
            assert (temp_dir_path / "model.pth").exists()
            module_names_list = module_names if isinstance(module_names, list) else [module_names]
            for module_name in module_names_list:
                module_dir = temp_dir_path / module_name
                files = {file.name for file in module_dir.glob("**/*")}
                assert "client.zip" in files and "server.zip" in files


@pytest.mark.use_gpu
# Dependency 'huggingface-hub' raises a 'FutureWarning' from version 0.23.0 when calling the
# 'from_pretrained' method
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "list_or_str_private_modules_names, expected_accuracy, has_pbs",
    [
        ("transformer.h.0.mlp", 0.95, True),
        (["transformer.h.0.mlp", "transformer.h.1.mlp"], 0.40, True),
        ("transformer.h.0.mlp.c_fc", 1.0, False),
    ],
)
@pytest.mark.parametrize("transformers_installed", [True, False])
@pytest.mark.parametrize("glwe_backend_installed", [True, False])
def test_gpt2_hybrid_mlp(
    list_or_str_private_modules_names,
    expected_accuracy,
    has_pbs,
    transformers_installed,
    glwe_backend_installed,
    monkeypatch,
    get_device,
    enforce_gpu_determinism,  # pylint: disable=unused-argument
):
    """Test GPT2 hybrid."""

    # Get GPT2 from Hugging Face
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(get_device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    prompt = "A long time ago,"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(get_device)

    # Run the test with using a single module in FHE
    assert isinstance(model, torch.nn.Module)
    run_hybrid_llm_test(
        model,
        input_ids,
        list_or_str_private_modules_names,
        expected_accuracy,
        has_pbs,
        transformers_installed,
        glwe_backend_installed,
        monkeypatch,
        get_device,
    )


@pytest.mark.use_gpu
@pytest.mark.filterwarnings("ignore:.*kthvalue CUDA does not have a deterministic implementation.*")
def test_hybrid_brevitas_qat_model(
    get_device, enforce_gpu_determinism
):  # pylint: disable=unused-argument
    """Test GPT2 hybrid."""
    n_bits = 3
    input_shape = 32
    output_shape = 4
    dataset_size = 100

    model = PartialQATModel(input_shape, output_shape, n_bits)
    inputs = torch.randn(
        (
            dataset_size,
            input_shape,
        )
    )
    # Run the test with using a single module in FHE
    model(inputs)
    assert isinstance(model, torch.nn.Module)
    hybrid_model = HybridFHEModel(model, module_names="sub_module")
    hybrid_model.compile_model(x=inputs, device=get_device)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Get the temp directory path
        hybrid_model.save_and_clear_private_info(temp_dir_path)

        # Check that files are there
        assert (temp_dir_path / "model.pth").exists()


# Dependency 'huggingface-hub' raises a 'FutureWarning' from version 0.23.0 when calling the
# 'from_pretrained' method
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_gpt2_hybrid_mlp_module_not_found():
    """Test GPT2 hybrid."""

    # Get GPT2 from Hugging Face
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Raise error module not found
    fake_module_name = "does_not_exist"
    with pytest.raises(ValueError, match=f"No module found for name {fake_module_name}"):
        HybridFHEModel(model, module_names=fake_module_name)


def test_invalid_model():
    """Test that a TypeError is raised when the model is not a torch.nn.Module."""

    # Create an invalid model (not a torch.nn.Module)
    invalid_model = "This_is_not_a_model"

    # Attempt to create a HybridFHEModel with an invalid model type and expect a TypeError
    with pytest.raises(TypeError, match="The model must be a PyTorch or Brevitas model."):
        HybridFHEModel(invalid_model, module_names="sub_module")


@pytest.mark.use_gpu
@pytest.mark.parametrize("use_dynamic_quantization, n_bits", [(True, 8), (False, 10)])
@pytest.mark.parametrize("n_hidden", [256, 512, 2048])
def test_hybrid_glwe_correctness(
    n_hidden, use_dynamic_quantization, n_bits, get_device, enforce_gpu_determinism
):  # pylint: disable=unused-argument
    """Tests that the GLWE backend produces correct results for the hybrid model."""

    num_samples = 200

    def prepare_data(x, y, test_size=0.1, random_state=42, device="cpu"):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)
        return x_train, x_test, y_train, y_test

    # Generate random data with n_hidden features and n_hidden classes
    # keeping input and output dimensions equal to n_hidden.
    x1_data = numpy.random.randn(num_samples, n_hidden)
    y1_data = numpy.random.randint(0, n_hidden, size=num_samples)  # n_hidden classes

    # Prepare data
    x1_train, x1_test, y1_train, y1_test = prepare_data(x1_data, y1_data, device=get_device)

    model = FCSmall(n_hidden, torch.nn.ReLU, hidden=n_hidden).to(get_device)
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 100
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x1_train)
        loss = torch.nn.functional.cross_entropy(outputs, y1_train)
        loss.backward()
        optimizer.step()

    model.eval()

    param_names = []
    for k, p in model.named_modules():
        if isinstance(p, torch.nn.Linear):
            param_names.append(k)

    y_torch = model(x1_test).detach().cpu().numpy()
    hybrid_local = HybridFHEModel(model, param_names)

    # This internal flag tells us whether all the layers
    # were linear and were replaced with the GLWE backend
    # Check if GLWE optimization should be used based on input dimension
    should_use_glwe = n_hidden >= 512
    is_pure_linear = hybrid_local._has_only_large_linear_layers  # pylint: disable=protected-access
    assert is_pure_linear == should_use_glwe

    hybrid_local.compile_model(
        x1_train,
        n_bits=n_bits,
        use_dynamic_quantization=use_dynamic_quantization,
        device=get_device,
    )

    y_qm = hybrid_local(x1_test, fhe="disable").cpu().numpy()
    y_hybrid_torch = hybrid_local(x1_test, fhe="torch").detach().cpu().numpy()

    # Only test GLWE execution if input dimension is >= 512
    if should_use_glwe:
        y_glwe = hybrid_local(x1_test, fhe="execute").cpu().numpy()

        y1_test = y1_test.cpu().numpy()
        n_correct_fp32 = numpy.sum(numpy.argmax(y_torch, axis=1) == y1_test)
        n_correct_qm = numpy.sum(numpy.argmax(y_qm, axis=1) == y1_test)
        n_correct_glwe = numpy.sum(numpy.argmax(y_glwe, axis=1) == y1_test)

        # These two should be exactly the same
        assert numpy.all(numpy.allclose(y_torch, y_hybrid_torch, rtol=1, atol=0.001))

        # The clear quantization vs fp32 test has more tolerance
        threshold_fhe = 0.1

        diff = numpy.abs(y_torch - y_glwe) > threshold_fhe
        if numpy.any(diff):
            print(f"Value discrepancy detected for GLWE backend, with epsilon={threshold_fhe}")
            print("Model output (torch fp32)", y_torch[diff])
            print("Model output (glwe)", y_glwe[diff])
            print("Model output (quantized clear)", y_qm[diff])

        assert numpy.all(numpy.allclose(y_qm, y_glwe, rtol=1, atol=threshold_fhe))
        assert numpy.all(numpy.allclose(y_torch, y_glwe, rtol=1, atol=threshold_fhe))

        n_correct_delta_threshold_fhe = 1
        # Check accuracy between fp32 and glwe
        assert numpy.abs(n_correct_fp32 - n_correct_glwe) <= n_correct_delta_threshold_fhe

        # Check accuracy between quantized and glwe
        assert numpy.abs(n_correct_qm - n_correct_glwe) <= n_correct_delta_threshold_fhe
    else:
        # For non-GLWE cases, just verify the torch outputs match
        assert numpy.all(numpy.allclose(y_torch, y_hybrid_torch, rtol=1, atol=0.001))
        assert numpy.all(numpy.allclose(y_qm, y_hybrid_torch, rtol=1, atol=0.1))
