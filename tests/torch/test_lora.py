# pylint: disable=redefined-outer-name

"""Tests for the LoraTraining class and related modules in lora.py."""

import sys
from collections import namedtuple
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from transformers import Conv1D as TransformerConv1D

from concrete.ml.torch.lora import (
    BackwardModuleLinear,
    CustomLinear,
    ForwardBackwardModule,
    ForwardModuleLinear,
    LoraTraining,
    get_remote_names,
)


class DummyConfig:
    """A dummy configuration class to mimic model config."""

    def __init__(self, model_type):
        self.model_type = model_type


class DummyBaseModel:
    """A dummy base model class to mimic base_model.model."""

    def __init__(self, model_type):
        self.model = DummyModel(model_type)


class DummyModel(torch.nn.Module):
    """A dummy model class to mimic the actual model."""

    def __init__(self, model_type):
        super().__init__()
        self.config = DummyConfig(model_type)

    @staticmethod
    def forward(x):
        """Dummy forward method."""
        return x


class DummyInferenceModel(torch.nn.Module):
    """A dummy inference model with various layers."""

    def __init__(self):
        super().__init__()
        self.base_model = DummyBaseModel("gpt2")
        self.linear1 = torch.nn.Linear(2, 2)
        self.conv1d = TransformerConv1D(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.lora_layer = torch.nn.Linear(2, 2)  # Layer with 'lora' in name
        self.lora_layer_name = "lora_layer"

    def forward(self, x, labels=None):
        """A simple forward method that returns logits or loss."""
        x = self.linear1(x)
        x = self.conv1d(x)
        x = self.linear2(x)
        x = self.lora_layer(x)
        logits = x
        if labels is not None:
            loss = ((logits - labels) ** 2).mean()
            Output = namedtuple("Output", ["loss"])
            return Output(loss=loss)
        return logits


@pytest.fixture
def base_inference_model():
    """Fixture for creating a DummyInferenceModel instance."""
    return DummyInferenceModel()


@pytest.fixture
def base_lora_training(base_inference_model):
    """Fixture for creating a LoraTraining instance."""
    return LoraTraining(base_inference_model)


@pytest.mark.parametrize("skip_first", [True, False])
def test_lora_training_replace_layers(base_lora_training, skip_first):
    """Test that LoraTraining replaces layers correctly."""
    original_linear1 = base_lora_training.inference_model.linear1
    original_lora_layer = base_lora_training.inference_model.lora_layer

    # Replace layers with custom layers
    base_lora_training.replace_layers_with_custom(
        base_lora_training.inference_model, skip_first=skip_first
    )

    inference_model = base_lora_training.inference_model

    if skip_first:
        # First eligible layer should be skipped
        assert inference_model.linear1 is original_linear1
    else:
        assert isinstance(inference_model.linear1, CustomLinear)

    # Check that other eligible layers are replaced
    assert isinstance(inference_model.conv1d, CustomLinear)
    assert isinstance(inference_model.linear2, CustomLinear)

    # 'lora' layers should not be replaced
    assert inference_model.lora_layer is original_lora_layer


@pytest.mark.parametrize(
    "training_args",
    [
        {"gradient_accumulation_steps": 2, "max_grad_norm": 1.0},  # dict
        SimpleNamespace(gradient_accumulation_steps=2, max_grad_norm=1.0),  # namespace
        None,  # None
    ],
)
def test_update_training_parameters(base_lora_training, training_args):
    """Test update_training_parameters with different types of training_args."""
    inference_model = base_lora_training.inference_model
    optimizer = SGD(inference_model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1)
    loss_fn = nn.MSELoss()

    base_lora_training.update_training_parameters(optimizer, lr_scheduler, loss_fn, training_args)

    assert base_lora_training.optimizer is optimizer
    assert base_lora_training.lr_scheduler is lr_scheduler
    assert base_lora_training.loss_fn is loss_fn

    if training_args is None:
        assert base_lora_training.gradient_accumulation_steps == 1  # Default
        assert base_lora_training.max_grad_norm is None  # Default
    else:
        assert base_lora_training.gradient_accumulation_steps == 2
        assert base_lora_training.max_grad_norm == 1.0


def test_lora_training_forward_loss_fn_none(base_lora_training):
    """Test the forward method when loss_fn is None."""
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    loss, _ = base_lora_training((x, y))

    expected_loss = (
        base_lora_training.inference_model(x, labels=y).loss
        / base_lora_training.gradient_accumulation_steps
    ).item()

    assert abs(loss.item() - expected_loss) < 1e-6


def test_lora_training_forward_with_loss_fn(base_lora_training):
    """Test the forward method when loss_fn is provided."""
    loss_fn = nn.MSELoss()
    base_lora_training.update_training_parameters(loss_fn=loss_fn)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    outputs = base_lora_training.inference_model(x)
    expected_loss = loss_fn(outputs, y) / base_lora_training.gradient_accumulation_steps

    loss, _ = base_lora_training((x, y))

    assert abs(loss.item() - expected_loss.item()) < 1e-6


def test_lora_training_forward_no_loss():
    """Test that LoraTraining raises ValueError when model does not return a loss."""

    class NoLossInferenceModel(DummyInferenceModel):
        """An inference model that does not return a loss."""

        def forward(self, x, labels=None):
            """Forward method that does not return loss."""
            Output = namedtuple("Output", ["something_else"])
            return Output(something_else=torch.tensor(1.0))

    no_loss_inference_model = NoLossInferenceModel()
    lora_training = LoraTraining(no_loss_inference_model)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    with pytest.raises(ValueError) as exc_info:
        lora_training((x, y))
    assert "The model did not return a loss" in str(exc_info.value)


@pytest.mark.parametrize("enable", [True, False])
def test_lora_training_toggle_calibrate(base_lora_training, enable):
    """Test the toggle_calibrate method."""
    base_lora_training.toggle_calibrate(enable)
    assert base_lora_training.calibrate == enable


@pytest.mark.parametrize("enable", [True, False])
def test_lora_training_toggle_run_optimizer(base_lora_training, enable):
    """Test the toggle_run_optimizer method."""
    base_lora_training.toggle_run_optimizer(enable)
    assert base_lora_training.run_optimizer == enable


def test_lora_training_forward_with_optimizer(base_lora_training):
    """Test the forward method when run_optimizer is True."""
    inference_model = base_lora_training.inference_model
    optimizer = SGD(inference_model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1)
    loss_fn = nn.MSELoss()
    base_lora_training.update_training_parameters(
        optimizer,
        lr_scheduler,
        loss_fn,
        SimpleNamespace(gradient_accumulation_steps=1, max_grad_norm=1.0),
    )
    base_lora_training.replace_layers_with_custom(
        base_lora_training.inference_model, skip_first=False
    )
    base_lora_training.toggle_run_optimizer(True)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    # Save initial parameters
    initial_params = {name: param.clone() for name, param in inference_model.named_parameters()}

    # Perform forward pass
    _, _ = base_lora_training((x, y))

    # Ensure that only parameters with "lora" in their name have been updated
    for name, param in inference_model.named_parameters():
        if "lora" in name:
            assert not torch.equal(
                initial_params[name], param
            ), f"Lora parameter {name} was not updated"
        else:
            assert torch.equal(
                initial_params[name], param
            ), f"Non-lora parameter {name} was unexpectedly updated"


def test_lora_training_forward_calibrate(base_lora_training):
    """Test the forward method when calibration is enabled."""
    inference_model = base_lora_training.inference_model
    base_lora_training.toggle_calibrate(True)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    _, _ = base_lora_training((x, y))

    # Ensure that gradients are zeroed
    for param in inference_model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0)


@pytest.mark.parametrize("weight_transposed", [False, True])
def test_forward_module_linear(weight_transposed):
    """Test ForwardModuleLinear."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias = torch.tensor([0.5, -0.5])
    module = ForwardModuleLinear(weight, bias, weight_transposed=weight_transposed)

    input_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    output = module(input_tensor)

    if weight_transposed:
        expected_output = input_tensor @ weight + bias
    else:
        expected_output = input_tensor @ weight.t() + bias

    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("weight_transposed", [False, True])
def test_backward_module_linear(weight_transposed):
    """Test BackwardModuleLinear."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    module = BackwardModuleLinear(weight, weight_transposed=weight_transposed)

    grad_output = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    grad_input = module(grad_output)

    if weight_transposed:
        expected_grad_input = grad_output @ weight.t()
    else:
        expected_grad_input = grad_output @ weight

    assert torch.allclose(grad_input, expected_grad_input)


@pytest.mark.parametrize("weight_transposed", [False, True])
def test_custom_linear(weight_transposed):
    """Test the CustomLinear module."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = torch.tensor([0.5, -0.5], requires_grad=True)
    module = CustomLinear(weight, bias, weight_transposed=weight_transposed)

    input_tensor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = module(input_tensor)

    if weight_transposed:
        expected_output = input_tensor @ weight + bias
    else:
        expected_output = input_tensor @ weight.t() + bias

    assert torch.allclose(output, expected_output)

    # Test backward
    output.sum().backward()
    if weight_transposed:
        expected_grad_input = torch.ones_like(output) @ weight.t()
    else:
        expected_grad_input = torch.ones_like(output) @ weight

    assert input_tensor.grad is not None and torch.allclose(input_tensor.grad, expected_grad_input)


@pytest.mark.parametrize("weight_transposed", [False, True])
def test_forward_backward_module(weight_transposed):
    """Test the ForwardBackwardModule."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias = torch.tensor([0.5, -0.5])
    forward_module = ForwardModuleLinear(weight, bias, weight_transposed=weight_transposed)
    backward_module = BackwardModuleLinear(weight, weight_transposed=weight_transposed)

    input_tensor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = ForwardBackwardModule.apply(input_tensor, forward_module, backward_module)

    if weight_transposed:
        expected_output = input_tensor @ weight + bias
        expected_grad_input = torch.ones_like(output) @ weight.t()
    else:
        expected_output = input_tensor @ weight.t() + bias
        expected_grad_input = torch.ones_like(output) @ weight

    assert torch.allclose(output, expected_output)

    # Test backward
    output.sum().backward()

    assert input_tensor.grad is not None and torch.allclose(input_tensor.grad, expected_grad_input)


def test_get_remote_names():
    """Test get_remote_names function."""

    class TestModel(torch.nn.Module):
        """Test model for get_remote_names test."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.conv1d = TransformerConv1D(10, 10)
            self.embedding = torch.nn.Embedding(10, 10)
            self.lm_head = torch.nn.Linear(10, 10)
            self.lora_layer = torch.nn.Linear(10, 10)
            self.lora_layer_name = "lora_layer"

        def forward(self, x):
            """Forward method."""
            return self.lm_head(self.linear(x))

    model = TestModel()

    lora_training = LoraTraining(model)
    remote_names = get_remote_names(lora_training)
    expected_names = [
        "inference_model.linear",
        "inference_model.conv1d.forward_module",
        "inference_model.conv1d.backward_module",
    ]

    assert set(remote_names) == set(expected_names)

    # Test with include_embedding_layers=True
    remote_names_with_embeddings = get_remote_names(lora_training, include_embedding_layers=True)
    expected_names_with_embeddings = [
        "inference_model.linear",
        "inference_model.conv1d.forward_module",
        "inference_model.conv1d.backward_module",
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4609
        "inference_model.embedding",
        "inference_model.lm_head.forward_module",
        "inference_model.lm_head.backward_module",
    ]
    assert set(remote_names_with_embeddings) == set(expected_names_with_embeddings)


def test_lora_without_transformers():
    """
    Test the lora.py module when the transformers library is not installed.
    """

    # Save the original transformers module if it's already imported
    transformers_original = sys.modules.get("transformers", None)

    # Mock the transformers import to simulate it being unavailable
    with mock.patch.dict("sys.modules", {"transformers": None}):
        # Reload the lora module to apply the mocked transformers import
        if "concrete.ml.torch.lora" in sys.modules:
            del sys.modules["concrete.ml.torch.lora"]
        import concrete.ml.torch.lora as lora  # pylint: disable=R0402,C0415

        # Ensure that TransformerConv1D is None
        assert lora.TransformerConv1D is None

        # Create a simple model without any Conv1D layers
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )

        # Initialize LoraTraining with the model
        lora_training = lora.LoraTraining(model)

        # Check that layers have been replaced with CustomLinear
        replaced_layers = []
        for name, module in lora_training.inference_model.named_modules():
            if isinstance(module, lora.CustomLinear):
                replaced_layers.append(name)

        # Assert that CustomLinear layers have been added
        assert len(replaced_layers) > 0, "No layers were replaced with CustomLinear."

        # Prepare input data
        x = torch.randn(3, 10)  # Batch size 3, input size 10
        y = torch.randint(0, 5, (3,))  # Batch size 3, number of classes 5

        # Define a simple loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Update training parameters
        lora_training.update_training_parameters(loss_fn=loss_fn)

        # Perform a forward pass
        loss, grad_norm = lora_training((x, y))

        # Check that loss is computed and gradients are updated
        assert loss.requires_grad, "Loss does not require gradients."
        assert loss.item() > 0, "Loss should be greater than zero."

        # Since optimizer is not set, grad_norm should be None
        assert grad_norm is None, "Gradient norm should be None when optimizer is not set."

    # Restore the original transformers module after the test
    if transformers_original is not None:
        sys.modules["transformers"] = transformers_original
    elif "transformers" in sys.modules:
        del sys.modules["transformers"]
