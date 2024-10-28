"""Tests for the hybrid model converter."""

import sys
import tempfile
from pathlib import Path
from typing import List, Union

import numpy
import pytest
import torch
from concrete.fhe import Configuration
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import concrete.ml.torch.hybrid_model
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


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def run_hybrid_llm_test(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    module_names: Union[str, List],
    expected_accuracy,
    has_pbs: bool,
    has_pbs_reshape: bool,
    monkeypatch,
    transformers_installed,
    glwe_backend_installed,
):
    """Run the test for any model with its private module names."""

    # Multi-parameter strategy is used in order to speed-up the FHE executions
    configuration = Configuration(
        single_precision=False,
        compress_input_ciphertexts=True,
    )

    logits_simulate = None

    with monkeypatch.context() as m:
        if not transformers_installed:
            m.setitem(sys.modules, "transformers", None)
            if has_pbs_reshape:
                has_pbs = True

        # Propagate glwe_backend_installed state being tested to constants of affected modules
        for affected_module in (
            concrete.ml.quantization.linear_op_glwe_backend,
            concrete.ml.torch.hybrid_model,
        ):
            m.setattr(affected_module, "_HAS_GLWE_BACKEND", glwe_backend_installed)

        # Create a hybrid model
        hybrid_model = HybridFHEModel(model, module_names)
        try:
            hybrid_model.compile_model(
                inputs,
                p_error=10e-40,  # compare precisely simulate and disable
                n_bits=9,
                rounding_threshold_bits=8,
                configuration=configuration,
            )
        except RuntimeError as error:
            # When reshaping adds PBSs we sometimes encounter NoParametersFound
            # when compiling. In this case we skip the rest since we can't simulate
            # without compilation.
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4183
            assert "NoParametersFound" in error.args[0]
            pytest.skip(error.args[0])

        # Check we can run the simulate locally
        if has_pbs or not glwe_backend_installed:
            logits_simulate = hybrid_model(inputs, fhe="simulate").logits
        else:
            with pytest.raises(AssertionError, match=".*fhe=simulate is not supported.*"):
                hybrid_model(inputs, fhe="simulate")

    if has_pbs:
        # Check for non-zero programmable bootstrapping
        for module in hybrid_model.private_q_modules.values():
            assert module.fhe_circuit.statistics["programmable_bootstrap_count"] > 0, (
                "Programmable bootstrap count should be greater than 0, "
                f"but found {module.fhe_circuit.statistics['programmable_bootstrap_count']}"
            )
    else:
        # Check for zero programmable bootstrapping
        for module in hybrid_model.private_q_modules.values():
            # The RemoteModule does not have a circuit if it was optimized
            # (in the case of pure linear remote modules)
            assert (
                not module.fhe_circuit
                or module.fhe_circuit.statistics["programmable_bootstrap_count"] == 0
            ), (
                "Programmable bootstrap count should be 0, "
                f"but found {module.fhe_circuit.statistics['programmable_bootstrap_count']}"
            )

    logits_disable = hybrid_model(inputs, fhe="disable").logits
    logits_original = hybrid_model(inputs, fhe="torch").logits

    # Compare the topk accuracy of the FHE simulate circuit vs. the original.
    k = 5

    # Check that the topk next tokens are similar for the different FHE modes
    # and the original model.

    # Get the topk indices for logits_disable and logits_simulate
    topk_disable = logits_disable.topk(k, dim=-1).indices
    topk_original = logits_original.topk(k, dim=-1).indices

    # Compute accuracy of disable and simulate by checking
    # how many labels correspond with the topk_original
    accuracy_disable = (topk_disable == topk_original).float().mean().item()
    # Ensure logits_disable and logits_original return the same output for the logits
    # Assert that both accuracy values are above the expected threshold
    assert (
        accuracy_disable >= expected_accuracy
    ), f"Disable accuracy {accuracy_disable:.4f} is below the expected {expected_accuracy:.4f}"

    if logits_simulate is not None:
        assert torch.allclose(logits_disable, logits_simulate, atol=1e-7), "Outputs do not match!"
        topk_simulate = logits_simulate.topk(k, dim=-1).indices
        accuracy_simulate = (topk_simulate == topk_original).float().mean().item()
        assert accuracy_simulate >= expected_accuracy, (
            f"Simulate accuracy {accuracy_simulate:.4f} is below "
            f"the expected {expected_accuracy:.4f}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Get the temp directory path

        if not has_pbs and glwe_backend_installed:
            # Deployment of GLWE backend hybrid models is not yet supported
            with pytest.raises(AttributeError, match="The quantized module is not compiled.*"):
                hybrid_model.save_and_clear_private_info(temp_dir_path)

        else:
            hybrid_model.save_and_clear_private_info(temp_dir_path)

            hybrid_model.set_fhe_mode("remote")

            # At this point, the hybrid model does not have
            # the parameters necessaryto run the module_names
            module_names = module_names if isinstance(module_names, list) else [module_names]

            # Check that files are there
            assert (temp_dir_path / "model.pth").exists()
            for module_name in module_names:
                module_dir_path = temp_dir_path / module_name
                module_dir_files = set(str(elt.name) for elt in module_dir_path.glob("**/*"))
                for file_name in ["client.zip", "server.zip"]:
                    assert file_name in module_dir_files


# Dependency 'huggingface-hub' raises a 'FutureWarning' from version 0.23.0 when calling the
# 'from_pretrained' method
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "list_or_str_private_modules_names, expected_accuracy, has_pbs, has_pbs_reshape",
    [
        ("transformer.h.0.mlp", 0.95, True, False),
        (["transformer.h.0.mlp", "transformer.h.1.mlp"], 0.40, True, False),
        ("transformer.h.0.mlp.c_fc", 1.0, False, True),
    ],
)
@pytest.mark.parametrize("transformers_installed", [True, False])
@pytest.mark.parametrize("glwe_backend_installed", [True, False])
def test_gpt2_hybrid_mlp(
    list_or_str_private_modules_names,
    expected_accuracy,
    has_pbs,
    has_pbs_reshape,
    transformers_installed,
    glwe_backend_installed,
    monkeypatch,
):
    """Test GPT2 hybrid."""

    # Get GPT2 from Hugging Face
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    prompt = "A long time ago,"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Run the test with using a single module in FHE
    assert isinstance(model, torch.nn.Module)
    run_hybrid_llm_test(
        model,
        input_ids,
        list_or_str_private_modules_names,
        expected_accuracy,
        has_pbs,
        has_pbs_reshape,
        monkeypatch,
        transformers_installed,
        glwe_backend_installed,
    )


def test_hybrid_brevitas_qat_model():
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
    hybrid_model.compile_model(x=inputs)

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


@pytest.mark.parametrize("n_hidden", [512, 2048])
def test_hybrid_glwe_correctness(n_hidden):
    """Tests that the GLWE backend produces correct results for the hybrid model."""

    num_samples = 200

    def prepare_data(x, y, test_size=0.1, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        return x_train, x_test, y_train, y_test

    # Generate synthetic 2D data
    x1_data, y1_data = make_moons(n_samples=num_samples, noise=0.2, random_state=42)

    # Prepare data
    x1_train, x1_test, y1_train, y1_test = prepare_data(x1_data, y1_data)

    model = FCSmall(2, torch.nn.ReLU, hidden=n_hidden)
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

    y_torch = model(x1_test).detach().numpy()
    hybrid_local = HybridFHEModel(model, param_names)

    # This internal flag tells us whether all the layers
    # were linear and were replaced with the GLWE backend
    assert hybrid_local._all_layers_are_pure_linear  # pylint: disable=protected-access

    hybrid_local.compile_model(x1_train, n_bits=10)

    y_qm = hybrid_local(x1_test, fhe="disable").numpy()
    y_hybrid_torch = hybrid_local(x1_test, fhe="torch").detach().numpy()
    y_glwe = hybrid_local(x1_test, fhe="execute").numpy()

    y1_test = y1_test.numpy()
    n_correct_fp32 = numpy.sum(numpy.argmax(y_torch, axis=1) == y1_test)
    n_correct_qm = numpy.sum(numpy.argmax(y_qm, axis=1) == y1_test)
    n_correct_glwe = numpy.sum(numpy.argmax(y_glwe, axis=1) == y1_test)

    # These two should be exactly the same
    assert numpy.all(numpy.allclose(y_torch, y_hybrid_torch, rtol=1, atol=0.001))

    # The clear quantization vs fp32 test has more tolerance
    threshold_fhe = 0.01

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
