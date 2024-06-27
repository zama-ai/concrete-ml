"""Tests for the hybrid model converter."""

import sys
import tempfile
from pathlib import Path
from typing import List, Union

import pytest
import torch
from concrete.fhe import Configuration
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from concrete.ml.pytest.torch_models import PartialQATModel
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


# pylint: disable=too-many-locals
def run_hybrid_llm_test(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    module_names: Union[str, List],
    expected_accuracy,
    has_pbs: bool,
    has_pbs_reshape: bool,
    monkeypatch,
    transformers_installed,
):
    """Run the test for any model with its private module names."""

    # Multi-parameter strategy is used in order to speed-up the FHE executions
    configuration = Configuration(
        single_precision=False,
        compress_input_ciphertexts=True,
    )

    with monkeypatch.context() as m:
        if not transformers_installed:
            m.setitem(sys.modules, "transformers", None)
            if has_pbs_reshape:
                has_pbs = True
        # Create a hybrid model
        hybrid_model = HybridFHEModel(model, module_names)
        try:
            hybrid_model.compile_model(
                inputs,
                p_error=0.1,
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
            assert module.fhe_circuit.statistics["programmable_bootstrap_count"] == 0, (
                "Programmable bootstrap count should be 0, "
                f"but found {module.fhe_circuit.statistics['programmable_bootstrap_count']}"
            )

    # Check we can run the simulate locally
    logits_simulate = hybrid_model(inputs, fhe="simulate").logits
    logits_disable = hybrid_model(inputs, fhe="disable").logits
    logits_original = hybrid_model(inputs, fhe="torch").logits

    # Ensure logits_disable and logits_original return the same output for the logits
    assert torch.allclose(logits_disable, logits_simulate, atol=1e-7), "Outputs do not match!"

    # Compare the topk accuracy of the FHE simulate circuit vs. the original.
    k = 5

    # Check that the topk next tokens are similar for the different FHE modes
    # and the original model.

    # Get the topk indices for logits_disable and logits_simulate
    topk_disable = logits_disable.topk(k, dim=-1).indices
    topk_simulate = logits_simulate.topk(k, dim=-1).indices
    topk_original = logits_original.topk(k, dim=-1).indices

    # Compute accuracy of disable and simulate by checking
    # how many labels correspond with the topk_original
    accuracy_disable = (topk_disable == topk_original).float().mean().item()
    accuracy_simulate = (topk_simulate == topk_original).float().mean().item()

    # Assert that both accuracy values are above the expected threshold
    assert (
        accuracy_disable >= expected_accuracy
    ), f"Disable accuracy {accuracy_disable:.4f} is below the expected {expected_accuracy:.4f}"
    assert (
        accuracy_simulate >= expected_accuracy
    ), f"Simulate accuracy {accuracy_simulate:.4f} is below the expected {expected_accuracy:.4f}"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Get the temp directory path
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
def test_gpt2_hybrid_mlp(
    list_or_str_private_modules_names,
    expected_accuracy,
    has_pbs,
    has_pbs_reshape,
    transformers_installed,
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
