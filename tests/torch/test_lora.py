"""Tests for the LoRA (Low-Rank Adaptation) functionality in the torch module."""

# pylint: disable=redefined-outer-name

import tempfile
from collections import UserDict
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
import torch
from peft import LoraConfig, get_peft_model
from sklearn.datasets import make_circles
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from concrete.ml.torch.hybrid_backprop_linear import (
    BackwardModuleLinear,
    CustomLinear,
    ForwardModuleLinear,
)
from concrete.ml.torch.lora import LoraTrainer, LoraTraining, get_remote_names, optimizer_to

# Dummy models and datasets for testing


class DummyLoRAModel(nn.Module):
    """Dummy LoRA model for testing."""

    def __init__(self):
        super().__init__()
        # Simulate LoRA layers by including 'lora_a' attribute
        self.lora_a = nn.Parameter(torch.randn(10, 10))
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x, labels=None):
        """Forward pass."""
        logits = self.linear2(torch.relu(self.linear1(x)))
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)
            return {"loss": loss}
        return {"logits": logits}


class DummyGLWELoRAModel(nn.Module):
    """Dummy LoRA model for testing."""

    def __init__(self):
        super().__init__()
        # Simulate LoRA layers by including 'lora_a' attribute
        self.lora_a = nn.Parameter(torch.randn(512, 512))
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)

    def forward(self, x, labels=None):
        """Forward pass."""
        logits = self.linear2(torch.relu(self.linear1(x)))
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)
            return {"loss": loss}
        return {"logits": logits}


class DummyLoRAModelNoLoss(nn.Module):
    """Dummy LoRA model without loss function for testing."""

    def __init__(self):
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(10, 10))
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        """Forward pass."""
        logits = self.linear2(torch.relu(self.linear1(x)))
        return {"logits": logits}


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        """Forward pass."""
        logits = self.linear2(torch.relu(self.linear1(x)))
        return {"logits": logits}


@pytest.fixture
def dummy_lora_model():
    """Dummy LoRA model for testing."""
    return DummyLoRAModel()


@pytest.fixture
def dummy_model():
    """Dummy model for testing."""
    return DummyModel()


def test_assert_has_lora_layers_with_lora_layers(dummy_lora_model):
    """Test assert_has_lora_layers with LoRA layers."""
    LoraTraining.assert_has_lora_layers(dummy_lora_model)


def test_assert_has_lora_layers_without_lora_layers(dummy_model):
    """Test assert_has_lora_layers without LoRA layers."""
    with pytest.raises(ValueError) as exc_info:
        LoraTraining.assert_has_lora_layers(dummy_model)
    assert "The model does not contain any detectable LoRA layers" in str(exc_info.value)


def test_replace_layers_with_custom():
    """Test replace_layers_with_custom."""
    model = DummyLoRAModel()
    n_layers_to_skip_for_backprop = 1
    LoraTraining.replace_layers_with_custom(model, n_layers_to_skip_for_backprop)
    # First linear layer should be skipped, second replaced
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, CustomLinear)


def test_replace_layers_with_custom_skips_lora_layers():
    """Test replace_layers_with_custom skips LoRA layers."""

    class ModelWithLoraLayer(nn.Module):
        """Model with LoRA layer for testing."""

        def __init__(self):
            super().__init__()
            self.lora_linear = nn.Linear(10, 10)
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            """Forward pass."""
            x = self.lora_linear(x)
            return self.linear(x)

    model = ModelWithLoraLayer()
    n_layers_to_skip_for_backprop = 0
    LoraTraining.replace_layers_with_custom(model, n_layers_to_skip_for_backprop)
    assert isinstance(model.lora_linear, nn.Linear)  # Should not be replaced
    assert isinstance(model.linear, CustomLinear)  # Should be replaced


def test_replace_layers_with_custom_recursive():
    """Test replace_layers_with_custom with nested modules."""

    class ModelWithNestedModules(nn.Module):
        """Model with nested modules for testing."""

        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        def forward(self, x):
            """Forward pass."""
            return self.layer1(x)

    model = ModelWithNestedModules()
    n_layers_to_skip_for_backprop = 0
    LoraTraining.replace_layers_with_custom(model, n_layers_to_skip_for_backprop)
    assert isinstance(model.layer1[0], CustomLinear)
    assert isinstance(model.layer1[1], nn.ReLU)  # Should not be replaced
    assert isinstance(model.layer1[2], CustomLinear)


def test_forward_with_loss_fn():
    """Test forward with loss function."""
    model = DummyLoRAModel()
    loss_fn = nn.MSELoss()
    lora_training = LoraTraining(model, loss_fn=loss_fn)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    with pytest.raises(AssertionError, match=".*must return either a Tensor.*dictionary.*"):
        lora_training((x, y))


def test_forward_without_loss_fn_model_returns_loss():
    """Test forward without loss function when model returns loss."""
    model = DummyLoRAModel()
    lora_training = LoraTraining(model)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    loss, _ = lora_training((x, y))
    assert isinstance(loss, torch.Tensor)


def test_forward_without_loss_fn_model_returns_loss_as_attribute():
    """Test forward without loss function when model returns loss as attribute."""

    class DummyLoRAModelReturnsObject(nn.Module):
        """Dummy LoRA model returning object with loss."""

        def __init__(self):
            super().__init__()
            self.lora_a = nn.Parameter(torch.randn(10, 10))
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)

        def forward(self, x, labels=None):
            """Forward pass."""
            logits = self.linear2(torch.relu(self.linear1(x)))

            class OutputObject:
                """Output object containing logits and optional loss."""

                def __init__(self, logits, loss=None):
                    self.logits = logits
                    self.loss = loss

            if labels is not None:
                loss = nn.functional.mse_loss(logits, labels)
                return OutputObject(logits, loss)
            return OutputObject(logits)

    model = DummyLoRAModelReturnsObject()
    lora_training = LoraTraining(model)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    loss, _ = lora_training((x, y))
    assert isinstance(loss, torch.Tensor)


def test_forward_with_less_than_two_inputs():
    """Test forward with less than two inputs."""
    model = DummyLoRAModel()
    lora_training = LoraTraining(model)
    x = torch.randn(5, 10)
    with pytest.raises(AssertionError) as exc_info:
        lora_training((x,))
    assert "must have two elements" in str(exc_info.value)


def test_set_loss_scaling_factor():
    """Test set_loss_scaling_factor."""
    model = DummyLoRAModel()
    lora_training = LoraTraining(model)
    lora_training.set_loss_scaling_factor(0.5)
    assert lora_training.loss_scaling_factor == 0.5


def test_lora_trainer_init():
    """Test LoraTrainer initialization."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lora_trainer = LoraTrainer(model, optimizer=optimizer)
    assert lora_trainer.lora_training_module is not None
    assert lora_trainer.hybrid_model is not None


@pytest.mark.parametrize(
    "model_cls", [DummyLoRAModel, DummyGLWELoRAModel], ids=["DummyLoRAModel", "DummyGLWELoRAModel"]
)
@pytest.mark.parametrize(
    "use_dynamic_quantization",
    [True, False],
    ids=["use_dynamic_quantization", "use_static_quantization"],
)
@pytest.mark.parametrize("fhe", ["disable", "execute"])
@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 5, 10),  # 3D input
        (2, 10),  # 2D input
    ],
    ids=["3d_input", "2d_input"],
)
def test_lora_trainer_train(model_cls, use_dynamic_quantization, fhe, input_shape):
    """Test LoraTrainer train."""
    model = model_cls()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    training_args = {"gradient_accumulation_steps": 1, "max_grad_norm": 1.0}
    lora_trainer = LoraTrainer(model, optimizer=optimizer, training_args=training_args)

    # Adjust last dimension for GLWE model
    if isinstance(model, DummyGLWELoRAModel):
        input_shape = (
            input_shape[:-1] + (512,) if len(input_shape) == 3 else input_shape[:-1] + (512,)
        )

    # Create input tensors with the same shape
    inputset = (torch.randn(*input_shape), torch.randn(*input_shape))

    # Compile the model
    lora_trainer.compile(inputset, use_dynamic_quantization=use_dynamic_quantization)

    # Create dataset with the same input shapes
    dataset = TensorDataset(torch.randn(*input_shape), torch.randn(*input_shape))
    train_loader = DataLoader(dataset, batch_size=1)
    lora_trainer.train(train_loader, num_epochs=1, fhe=fhe)


def test_lora_trainer_train_with_lr_scheduler():
    """Test LoraTrainer train with lr_scheduler."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = MagicMock()
    training_args = {"gradient_accumulation_steps": 1, "max_grad_norm": 1.0}
    lora_trainer = LoraTrainer(
        model, optimizer=optimizer, lr_scheduler=lr_scheduler, training_args=training_args
    )
    # Mock the hybrid_model's __call__ method
    lora_trainer.hybrid_model = MagicMock(
        return_value=(torch.tensor(1.0, requires_grad=True), None)
    )
    # Create dummy data loader
    dataset = TensorDataset(torch.randn(2, 5, 10), torch.randn(2, 5, 10))
    train_loader = DataLoader(dataset, batch_size=1)
    lora_trainer.train(train_loader, num_epochs=1)
    # Check that lr_scheduler.step() was called
    assert lr_scheduler.step.call_count > 0


def test_lora_trainer_save_and_clear_private_info():
    """Test LoraTrainer save_and_clear_private_info."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lora_trainer = LoraTrainer(model, optimizer=optimizer, loss_fn=nn.MSELoss())
    lora_trainer.hybrid_model.save_and_clear_private_info = MagicMock()
    lora_trainer.save_and_clear_private_info("path/to/model")
    lora_trainer.hybrid_model.save_and_clear_private_info.assert_called_once_with("path/to/model")


def test_custom_linear_forward_backward():
    """Test CustomLinear forward and backward."""
    weight = torch.randn(20, 10)
    bias = torch.randn(20)
    custom_linear = CustomLinear(weight, bias)
    x = torch.randn(5, 10, requires_grad=True)
    y = custom_linear(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_custom_linear_weight_transposed():
    """Test CustomLinear with weight transposed."""
    weight = torch.randn(10, 20)
    bias = torch.randn(20)
    custom_linear = CustomLinear(weight, bias, weight_transposed=True)
    x = torch.randn(5, 10, requires_grad=True)
    y = custom_linear(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_get_remote_names():
    """Test get_remote_names."""
    model = DummyLoRAModel()
    LoraTraining.replace_layers_with_custom(model, n_layers_to_skip_for_backprop=0)
    remote_names = get_remote_names(model)
    assert "linear1.forward_module" in remote_names
    assert "linear1.backward_module" in remote_names
    assert "linear2.forward_module" in remote_names
    assert "linear2.backward_module" in remote_names
    assert "lora_a" not in remote_names


def test_get_remote_names_include_embedding_layers():
    """Test get_remote_names with include_embedding_layers."""

    class ModelWithEmbedding(nn.Module):
        """Model with embedding layer for testing."""

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 10)
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            """Forward pass."""
            x = self.embedding(x)
            x = self.linear(x)
            return x

    model = ModelWithEmbedding()
    remote_names = get_remote_names(model, include_embedding_layers=True)
    assert "embedding" in remote_names
    assert "linear" in remote_names


def test_get_remote_names_skips_lm_head_when_excluded():
    """Test get_remote_names skips lm_head when excluded."""

    class ModelWithLMHead(nn.Module):
        """Model with lm_head for testing."""

        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            """Forward pass."""
            return self.linear(x)

    model = ModelWithLMHead()
    remote_names = get_remote_names(model, include_embedding_layers=False)
    assert "lm_head" not in remote_names
    assert "linear" in remote_names


def test_replace_layers_with_transformer_conv1d(monkeypatch):
    """Test replace_layers_with_custom with TransformerConv1D."""

    class MockTransformerConv1D(nn.Module):
        """Mock TransformerConv1D module for testing."""

        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.randn(out_features))

        def forward(self, x):
            """Forward pass."""
            return x @ self.weight.t() + self.bias

    # Patch TransformerConv1D and LINEAR_LAYERS in the lora module
    monkeypatch.setattr("concrete.ml.torch.lora.TransformerConv1D", MockTransformerConv1D)
    monkeypatch.setattr("concrete.ml.torch.lora.LINEAR_LAYERS", (nn.Linear, MockTransformerConv1D))

    class ModelWithConv1D(nn.Module):
        """Model with Conv1D layer for testing."""

        def __init__(self):
            super().__init__()
            self.conv1d = MockTransformerConv1D(10, 10)

        def forward(self, x):
            """Forward pass."""
            return self.conv1d(x)

    model = ModelWithConv1D()
    n_layers_to_skip_for_backprop = 0
    LoraTraining.replace_layers_with_custom(model, n_layers_to_skip_for_backprop)
    assert isinstance(model.conv1d, CustomLinear)


def test_forward_backward_module():
    """Test the ForwardBackwardModule autograd function."""
    weight = torch.randn(20, 10)
    bias = torch.randn(20)
    forward_module = ForwardModuleLinear(weight, bias)
    backward_module = BackwardModuleLinear(weight)
    x = torch.randn(5, 10)
    y = forward_module(x)
    grad_output = torch.randn_like(y)
    grad_input = backward_module(grad_output)
    assert grad_input.shape == x.shape


def test_lora_training_forward_with_loss_fn_and_attention_mask():
    """Test LoraTraining forward using a custom loss_fn and attention_mask."""

    class ModelWithAttention(nn.Module):
        """Model that supports attention_mask for testing."""

        def __init__(self):
            super().__init__()
            self.lora_a = nn.Parameter(torch.randn(10, 10))
            self.linear = nn.Linear(10, 10)

        def forward(self, x, attention_mask=None):
            """Forward pass."""
            if attention_mask is not None:
                return {"logits": self.linear(x + attention_mask)}
            return {"logits": self.linear(x)}

    # Define a simple loss function
    def simple_loss_fn(logits, labels):
        return nn.MSELoss()(logits, labels)

    model = ModelWithAttention()

    # Instantiate LoraTraining with a custom loss_fn
    lora_training = LoraTraining(model, loss_fn=simple_loss_fn)

    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    attention_mask = torch.randint(0, 2, (5, 10))

    # Call forward with (input_ids, labels, attention_mask)
    loss, _ = lora_training({"x": x, "labels": y, "attention_mask": attention_mask})
    assert isinstance(loss, torch.Tensor)


def test_lora_training_forward_with_additional_inputs():
    """Test LoraTraining forward with additional inputs."""

    class ModelWithAttention(nn.Module):
        """Model with attention input for testing."""

        def __init__(self):
            super().__init__()
            self.lora_a = nn.Parameter(torch.randn(10, 10))
            self.linear = nn.Linear(10, 10)

        def forward(self, x, attention_mask=None, labels=None):
            """Forward pass with an attention mask."""
            # Just treat the attention_mask as an extra input
            # and add it to x before passing through linear.
            if attention_mask is not None:
                logits = self.linear(x + attention_mask)
            else:
                logits = self.linear(x)

            if labels is not None:
                loss = nn.functional.mse_loss(logits, labels)
                return {"loss": loss}
            return {"logits": logits}

    model = ModelWithAttention()
    lora_training = LoraTraining(model)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    attention_mask = torch.randint(0, 2, (5, 10))

    loss, _ = lora_training({"x": x, "labels": y, "attention_mask": attention_mask})
    assert isinstance(loss, torch.Tensor)


def test_lora_training_forward_with_no_loss_fn_and_no_labels():
    """Test LoraTraining when model returns loss=None and no loss_fn provided."""
    model = DummyLoRAModel()
    lora_training = LoraTraining(model)
    x = torch.randn(5, 10)
    y = None  # No labels provided
    with pytest.raises(ValueError) as exc_info:
        lora_training((x, y))
    assert "The model did not return a loss." in str(exc_info.value)


@pytest.mark.parametrize("has_loss", [True, False])
def test_lora_trainer_train_with_various_batch_types(has_loss):
    """Test LoraTrainer.train with batches of different types."""
    model = DummyLoRAModel()
    loss_fn = nn.functional.mse_loss if has_loss else None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lora_trainer = LoraTrainer(model, optimizer=optimizer, loss_fn=loss_fn)

    class DictDataset(Dataset):
        """Dataset with dict items."""

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    class ListDataset(Dataset):
        """Dataset with list items."""

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Test with dict batch
    dataset_dict = [{"x": torch.randn(5, 10), "labels": torch.randn(5, 10)} for _ in range(2)]
    train_loader_dict: DataLoader = DataLoader(DictDataset(dataset_dict), batch_size=1)
    lora_trainer.compile({"x": torch.randn(1, 5, 10), "labels": torch.randn(1, 5, 10)}, n_bits=8)
    lora_trainer.train(train_loader_dict, num_epochs=1, fhe="disable")

    # Test with UserDict batch
    dataset_user_dict = [
        UserDict({"x": torch.randn(5, 10), "labels": torch.randn(5, 10)}) for _ in range(2)
    ]
    train_loader_udict: DataLoader = DataLoader(DictDataset(dataset_user_dict), batch_size=1)
    lora_trainer.train(train_loader_udict, num_epochs=1, fhe="disable")

    if not has_loss:
        # Test with list/tuple batch
        dataset_list = [(torch.randn(5, 10), torch.randn(5, 10)) for _ in range(2)]
        train_loader_list: DataLoader = DataLoader(ListDataset(dataset_list), batch_size=1)
        lora_trainer.train(train_loader_list, num_epochs=1, fhe="disable")


def test_lora_trainer_inference_keygen():
    """Test LoraTrainer on a large model and infer with FHE."""
    model = DummyGLWELoRAModel()
    loss_fn = None  # nn.functional.mse_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lora_trainer = LoraTrainer(model, optimizer=optimizer, loss_fn=loss_fn)

    # Test with list/tuple batch
    dataset_list = [(torch.randn(5, 512), torch.randn(5, 512)) for _ in range(2)]
    lora_trainer.compile((dataset_list[0][0], dataset_list[0][1]))
    lora_trainer.hybrid_model((dataset_list[0][0], dataset_list[0][1]), fhe="execute")


def test_lora_move_optimizer():
    """Test LoraTrainer.train with batches of different types."""
    model = DummyLoRAModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    class ListDataset(Dataset):
        """Dataset with list items."""

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset_list = [(torch.randn(5, 10), torch.randn(5, 10)) for _ in range(200)]
    train_loader_list: DataLoader = DataLoader(ListDataset(dataset_list), batch_size=1)

    model.train()
    optimizer.zero_grad()
    for _, batch in enumerate(train_loader_list):
        res = model(batch[0])["logits"]
        loss = torch.nn.functional.mse_loss(res, batch[1])
        loss.backward()
        optimizer.step()

    optimizer_to(optimizer, "cpu")


def test_lora_trainer_train_with_gradient_accumulation():
    """Test LoraTrainer.train with gradient accumulation steps."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    training_args = {"gradient_accumulation_steps": 2, "max_grad_norm": 1.0}
    lora_trainer = LoraTrainer(model, optimizer=optimizer, training_args=training_args)
    # Mock the hybrid_model's __call__ method
    lora_trainer.hybrid_model = MagicMock(
        return_value=(torch.tensor(1.0, requires_grad=True), None)
    )
    # Create dummy data loader
    dataset = TensorDataset(torch.randn(4, 5, 10), torch.randn(4, 5, 10))
    train_loader: DataLoader = DataLoader(dataset, batch_size=1)
    lora_trainer.train(train_loader, num_epochs=1)


def test_get_remote_names_with_lora_in_name():
    """Test get_remote_names skips modules with 'lora' in name."""

    class ModelWithLoraInName(nn.Module):
        """Model with LoRA layer for testing."""

        def __init__(self):
            super().__init__()
            self.lora_linear = nn.Linear(10, 10)
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            """Forward pass with lora_linear."""
            x = self.lora_linear(x)
            x = self.linear(x)
            return x

    model = ModelWithLoraInName()
    remote_names = get_remote_names(model)
    assert "lora_linear" not in remote_names
    assert "linear" in remote_names


def test_lora_training_init_validates_model_signature():
    """Test LoraTraining initialization validates model's forward signature."""

    class ModelWithoutLabels(nn.Module):
        """Model without labels parameter in forward."""

        def __init__(self):
            super().__init__()
            self.lora_a = nn.Parameter(torch.randn(10, 10))
            self.linear = nn.Linear(10, 10)

        def forward(self, x):  # No labels parameter
            """Forward pass without labels parameter."""
            return {"logits": self.linear(x)}

    model = ModelWithoutLabels()

    with pytest.raises(ValueError) as exc_info:
        LoraTraining(model, loss_fn=None)  # No loss_fn provided
    assert "must accept a 'labels' parameter" in str(exc_info.value)


def test_lora_train_mlp():
    """Test that a MLP can be lora trained using FHE."""

    class SimpleMLP(nn.Module):
        """Simple MLP model without LoRA layers."""

        def __init__(self, input_size=2, hidden_size=128, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x, labels=None):  # pylint: disable=unused-argument
            """Forward pass of the MLP."""
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # Create an initial model
    model = SimpleMLP()

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=1, lora_alpha=1, lora_dropout=0.01, target_modules=["fc1", "fc2"], bias="none"
    )

    peft_model = get_peft_model(model, lora_config)

    # Generate a second data-set for demonstration purposes
    x_task2, y_task2 = make_circles(n_samples=32, noise=0.2, factor=0.5)
    train_loader_task2 = DataLoader(
        TensorDataset(torch.Tensor(x_task2), torch.LongTensor(y_task2)), batch_size=32, shuffle=True
    )

    # Define a simple loss function
    def simple_loss(outputs, targets):
        return torch.nn.functional.cross_entropy(outputs, targets)

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-3)

    # Initialize trainer with the loss and optimizer
    lora_trainer = LoraTrainer(
        peft_model,
        optimizer=optimizer,
        loss_fn=simple_loss,
    )
    # Build a representative data-set for compilation
    inputset = (
        torch.Tensor(x_task2[:16]),
        torch.LongTensor(y_task2[:16]),
    )

    # Calibrate and compile the model with 8-bit quantization
    lora_trainer.compile(inputset, n_bits=8)
    # Train in FHE mode
    lora_trainer.train(train_loader_task2, fhe="execute")


def test_checkpoint_saving_loading_and_resuming():
    """Test checkpoint saving, loading, resuming training and result tracking."""
    num_samples = 10
    input_data = torch.randn(num_samples, 10)
    target_data = torch.randn(num_samples, 10)
    dataset = TensorDataset(input_data, target_data)
    train_loader = DataLoader(dataset, batch_size=1)

    # Create a dummy model and optimizer.
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Add a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # Create an arbitrary input set for model compilation.
    inputset = (torch.randn(5, 10), torch.randn(5, 10))

    # Use a temporary directory for checkpoint files.
    with tempfile.TemporaryDirectory() as temp_dir:
        # First training phase
        lora_trainer = LoraTrainer(
            model, optimizer=optimizer, lr_scheduler=lr_scheduler, checkpoint_dir=temp_dir
        )
        lora_trainer.compile(inputset, n_bits=8)
        lora_trainer.train(train_loader, fhe="disable", num_epochs=3, device="cpu")

        # Check that a checkpoint file is created for each epoch.
        for epoch in range(1, 4):
            checkpoint_path = Path(temp_dir) / f"checkpoint_epoch_{epoch}.pth"
            assert checkpoint_path.exists(), f"Checkpoint for epoch {epoch} not found."

        # Test that training losses have been recorded.
        initial_losses = lora_trainer.get_training_losses()
        assert len(initial_losses) > 0, "No training losses recorded in the first phase."

        # Test that gradient statistics have been recorded.
        grad_stats = lora_trainer.get_gradient_stats()
        assert isinstance(grad_stats, dict), "Gradient stats should be a dictionary."
        for param_name, stats in grad_stats.items():
            assert len(stats) > 0, f"No gradient stats recorded for parameter '{param_name}'."

        # Test loading non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            lora_trainer.load_checkpoint("non_existent_checkpoint.pth")

        # Store initial learning rate
        initial_lr = optimizer.param_groups[0]["lr"]

        # New model which undergoes same layer replacement as original model.
        # This is needed to ensure that the loaded checkpoint is compatible with the new model
        model = DummyLoRAModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        resumed_trainer = LoraTrainer(
            model, optimizer=optimizer, lr_scheduler=lr_scheduler, checkpoint_dir=temp_dir
        )

        # Compile the model, which will automatically load the latest checkpoint
        loaded_epoch, loaded_global_step = resumed_trainer.compile(inputset, n_bits=8)

        # Verify the loaded checkpoint is from epoch 3 (the latest)
        assert loaded_epoch == 3, "Latest checkpoint epoch does not match expected value."
        assert loaded_global_step > 0, "Loaded global step should be greater than 0."

        # Verify that the learning rate was properly restored from checkpoint
        loaded_lr = optimizer.param_groups[0]["lr"]
        expected_lr = initial_lr * (lr_scheduler.gamma**loaded_epoch)
        assert (
            abs(loaded_lr - expected_lr) < 1e-6
        ), "Learning rate was not properly restored from checkpoint"

        # Resume training for 1 additional epoch
        resumed_trainer.train(
            train_loader,
            fhe="disable",
            num_epochs=4,  # Note: training will start from epoch (loaded_epoch + 1)=4
            device="cpu",
        )

        # After resuming, check that additional training losses have been recorded.
        resumed_losses = resumed_trainer.get_training_losses()
        assert len(resumed_losses) > len(
            initial_losses
        ), "No additional training losses recorded after resuming."

        # Also check that gradient statistics have been updated.
        resumed_grad_stats = resumed_trainer.get_gradient_stats()
        for param_name, stats in resumed_grad_stats.items():
            assert (
                len(stats) > 0
            ), f"No gradient stats recorded for parameter '{param_name}' after resuming."


def test_evaluation():
    """Test that evaluation is done correctly."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = DataLoader(TensorDataset(torch.randn(5, 10), torch.randn(5, 10)), batch_size=1)
    eval_loader = DataLoader(TensorDataset(torch.randn(3, 10), torch.randn(3, 10)), batch_size=1)
    inputset = (torch.randn(5, 10), torch.randn(5, 10))

    # Define a simple evaluation metric function with a counter to track calls
    eval_calls: List[int] = []

    def eval_metric_fn(model, loader):
        # Verify model is in eval mode during evaluation
        assert not model.training, "Model should be in eval mode during evaluation"
        eval_calls.append(len(eval_calls) + 1)  # Track call count

        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                output = model(x)
                # Extract logits from the output dictionary
                logits = output["logits"]
                loss = torch.nn.functional.mse_loss(logits, y)
                total_loss += loss.item()
        return {"eval_loss": total_loss / len(loader)}

    # Test case 1: With evaluation data and metric function
    lora_trainer = LoraTrainer(
        model,
        optimizer=optimizer,
        eval_loader=eval_loader,
        eval_metric_fn=eval_metric_fn,
        eval_steps=2,
    )

    lora_trainer.compile(inputset, n_bits=8)
    # Train for a few steps to trigger evaluation
    lora_trainer.train(train_loader, num_epochs=1, fhe="disable")

    # Verify evaluations occurred
    assert len(eval_calls) > 0, "No evaluations were performed"
    assert eval_calls == list(range(1, len(eval_calls) + 1)), "Evaluations were not sequential"

    # Verify model is back in training mode after training
    assert model.training, "Model should be back in training mode after training"

    # Test case 2: Without evaluation data and metric function
    eval_calls.clear()  # Reset counter
    # New model without evaluation data and metric function
    model_no_eval = DummyLoRAModel()
    optimizer_no_eval = torch.optim.SGD(model_no_eval.parameters(), lr=0.01)
    lora_trainer_no_eval = LoraTrainer(model_no_eval, optimizer=optimizer_no_eval)

    lora_trainer_no_eval.compile(inputset, n_bits=8)
    # Train for a few steps
    lora_trainer_no_eval.train(train_loader, num_epochs=1, fhe="disable")

    # Verify no evaluations occurred
    msg = "Evaluations should not occur without eval_loader and eval_metric_fn"
    assert len(eval_calls) == 0, msg
