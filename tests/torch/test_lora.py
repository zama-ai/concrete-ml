"""Tests for the LoraTraining class and related modules in lora.py."""

from collections import namedtuple

import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from concrete.ml.torch.lora import (
    BackwardModule,
    CustomConv1D,
    CustomLinear,
    ForwardBackwardModule,
    ForwardModule,
    LoraTraining,
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
    """A dummy inference model."""

    def __init__(self, model_type):
        super().__init__()
        self.base_model = DummyBaseModel(model_type)
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x, labels=None):
        """A simple forward method that returns a loss."""
        logits = self.linear(x)
        loss = ((logits - labels) ** 2).mean() if labels is not None else logits.mean()
        Output = namedtuple("Output", ["loss"])
        return Output(loss=loss)


def test_lora_training_init_supported_model():
    """Test that LoraTraining initializes correctly with a supported model type."""
    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)
    assert lora_training.inference_model is inference_model
    assert lora_training.gradient_accumulation_steps == 2


def test_lora_training_init_unsupported_model():
    """Test that LoraTraining raises ValueError with an unsupported model type."""
    inference_model = DummyInferenceModel("bert")
    with pytest.raises(ValueError) as exc_info:
        LoraTraining(inference_model, gradient_accumulation_steps=2)
    assert "Unsupported model type" in str(exc_info.value)


def test_lora_training_forward():
    """Test the forward method of LoraTraining."""
    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

    loss, grad_norm = lora_training((x, y))
    expected_loss = ((inference_model.linear(x) - y) ** 2).mean().item() / 2
    assert loss.item() == expected_loss
    assert grad_norm is None  # Since run_optimizer is False by default


def test_lora_training_forward_no_loss():
    """Test that LoraTraining raises ValueError when model does not return a loss."""

    class NoLossInferenceModel(DummyInferenceModel):
        """An inference model that does not return a loss."""

        def forward(self, x, labels=None):
            Output = namedtuple("Output", ["something_else"])
            return Output(something_else=torch.tensor(1.0))

    inference_model = NoLossInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    with pytest.raises(ValueError) as exc_info:
        lora_training((x, y))
    assert "The model did not return a loss" in str(exc_info.value)


def test_lora_training_toggle_calibrate():
    """Test the toggle_calibrate method of LoraTraining."""
    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    assert not lora_training.calibrate
    lora_training.toggle_calibrate(True)
    assert lora_training.calibrate
    lora_training.toggle_calibrate(False)
    assert not lora_training.calibrate


def test_lora_training_toggle_run_optimizer():
    """Test the toggle_run_optimizer method of LoraTraining."""
    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    assert not lora_training.run_optimizer
    lora_training.toggle_run_optimizer(True)
    assert lora_training.run_optimizer
    lora_training.toggle_run_optimizer(False)
    assert not lora_training.run_optimizer


def test_lora_training_update_training_parameters():
    """Test the update_training_parameters method of LoraTraining."""
    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    optimizer = SGD(inference_model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1)
    TrainingArgs = namedtuple("TrainingArgs", ["gradient_accumulation_steps", "max_grad_norm"])
    training_args = TrainingArgs(2, 1.0)

    lora_training.update_training_parameters(optimizer, lr_scheduler, training_args)

    assert lora_training.optimizer is optimizer
    assert lora_training.lr_scheduler is lr_scheduler
    assert lora_training.max_grad_norm == training_args.max_grad_norm


def test_lora_training_forward_with_optimizer():
    """Test the forward method with optimizer enabled."""

    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    optimizer = SGD(inference_model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1)
    TrainingArgs = namedtuple("TrainingArgs", ["gradient_accumulation_steps", "max_grad_norm"])
    training_args = TrainingArgs(2, 1.0)

    lora_training.update_training_parameters(optimizer, lr_scheduler, training_args)
    lora_training.toggle_run_optimizer(True)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    # Compute expected_loss before the forward pass
    with torch.no_grad():
        expected_loss = ((inference_model.linear(x) - y) ** 2).mean().item() / 2

    # Save the initial parameters
    initial_params = [param.clone() for param in inference_model.parameters()]

    # Perform the forward pass
    loss, grad_norm = lora_training((x, y))

    # Assert that the loss is close to the expected loss
    assert abs(loss.item() - expected_loss) < 1e-6
    assert grad_norm is not None

    # Check that parameters have been updated by the optimizer
    for initial_param, param in zip(initial_params, inference_model.parameters()):
        assert not torch.equal(initial_param, param)


def test_forward_module():
    """Test the ForwardModule."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias = torch.tensor([0.5, -0.5])
    module = ForwardModule(weight, bias)

    input_tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    output = module(input_tensor)

    expected_output = input_tensor @ weight + bias
    assert output is not None and torch.allclose(output, expected_output)


def test_backward_module():
    """Test the BackwardModule."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    module = BackwardModule(weight)

    grad_output = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    grad_input = module(grad_output)

    expected_grad_input = grad_output @ weight.t()
    assert grad_input is not None and torch.allclose(grad_input, expected_grad_input)


def test_custom_conv1d():
    """Test the CustomConv1D module."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = torch.tensor([0.5, -0.5], requires_grad=True)
    module = CustomConv1D(weight, bias)

    input_tensor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = module(input_tensor)

    expected_output = input_tensor @ weight + bias
    assert output is not None and torch.allclose(output, expected_output)

    # Test backward pass
    output.sum().backward()
    expected_grad_input = torch.ones_like(output) @ weight.t()
    assert input_tensor.grad is not None and torch.allclose(input_tensor.grad, expected_grad_input)


def test_custom_linear():
    """Test the CustomLinear module."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = torch.tensor([0.5, -0.5], requires_grad=True)
    module = CustomLinear(weight, bias)

    input_tensor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = module(input_tensor)

    expected_output = input_tensor @ weight + bias
    assert output is not None and torch.allclose(output, expected_output)

    # Test backward pass
    output.sum().backward()
    expected_grad_input = torch.ones_like(output) @ weight.t()
    assert input_tensor.grad is not None and torch.allclose(input_tensor.grad, expected_grad_input)


def test_forward_backward_module():
    """Test the ForwardBackwardModule."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = torch.tensor([0.5, -0.5])
    forward_module = ForwardModule(weight, bias)
    backward_module = BackwardModule(weight)

    input_tensor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = ForwardBackwardModule.apply(input_tensor, forward_module, backward_module)

    expected_output = input_tensor @ weight + bias
    assert output is not None and torch.allclose(output, expected_output)

    # Test backward pass
    output.sum().backward()
    expected_grad_input = torch.ones_like(output) @ weight.t()
    assert input_tensor.grad is not None and torch.allclose(input_tensor.grad, expected_grad_input)


def test_lora_training_invalid_inference_model():
    """Test that LoraTraining raises ValueError when inference_model lacks required attributes."""

    # Create an inference model that lacks base_model
    class InvalidInferenceModel(torch.nn.Module):
        """An invalid inference model without base_model attribute."""

        @staticmethod
        def forward(x):
            """Dummy forward method."""
            return x

    inference_model = InvalidInferenceModel()
    with pytest.raises(ValueError) as exc_info:
        LoraTraining(inference_model, gradient_accumulation_steps=2)
    assert "Unable to determine the base model type." in str(exc_info.value)


def test_lora_training_forward_calibrate():
    """Test the forward method when calibration is enabled."""
    inference_model = DummyInferenceModel("gpt2")
    lora_training = LoraTraining(inference_model, gradient_accumulation_steps=2)

    # Enable calibration
    lora_training.toggle_calibrate(True)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[0.5, 1.5]])

    # Perform the forward pass
    loss, grad_norm = lora_training((x, y))

    # Since calibrate is True, grad_norm should be None
    assert grad_norm is None

    # Ensure that loss is computed correctly
    expected_loss = ((inference_model.linear(x) - y) ** 2).mean().item() / 2
    assert abs(loss.item() - expected_loss) < 1e-6

    # Ensure that gradients have been cleared (zeroed)
    for param in inference_model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0)
