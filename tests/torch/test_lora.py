"""Tests for the LoRA (Low-Rank Adaptation) functionality in the torch module."""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from concrete.ml.torch.hybrid_backprop_linear import (
    BackwardModuleLinear,
    CustomLinear,
    ForwardModuleLinear,
)
from concrete.ml.torch.lora import LoraTrainer, LoraTraining, get_remote_names

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
    loss, _ = lora_training((x, y))
    assert isinstance(loss, torch.Tensor)


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
    assert "Expected at least two inputs" in str(exc_info.value)


def test_toggle_calibrate():
    """Test toggle_calibrate."""
    model = DummyLoRAModel()
    lora_training = LoraTraining(model)
    lora_training.toggle_calibrate(True)
    assert lora_training.calibrate is True
    lora_training.toggle_calibrate(False)
    assert lora_training.calibrate is False


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


def test_lora_trainer_compile():
    """Test LoraTrainer compile."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lora_trainer = LoraTrainer(model, optimizer=optimizer)
    inputset = [(torch.randn(5, 10), torch.randn(5, 10))]
    # Mock the compile_model method
    lora_trainer.hybrid_model.compile_model = MagicMock()
    lora_trainer.compile(inputset)
    lora_trainer.hybrid_model.compile_model.assert_called_once()
    assert lora_trainer.lora_training_module.calibrate is False


def test_lora_trainer_train():
    """Test LoraTrainer train."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    training_args = {"gradient_accumulation_steps": 1, "max_grad_norm": 1.0}
    lora_trainer = LoraTrainer(model, optimizer=optimizer, training_args=training_args)
    # Mock the hybrid_model's __call__ method
    lora_trainer.hybrid_model = MagicMock(
        return_value=(torch.tensor(1.0, requires_grad=True), None)
    )
    # Create dummy data loader with different batch types
    dataset = TensorDataset(torch.randn(2, 5, 10), torch.randn(2, 5, 10))
    train_loader = DataLoader(dataset, batch_size=1)
    lora_trainer.train(train_loader, num_epochs=1, fhe="disable")


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
    attention_mask = torch.randn(5, 10)

    # Call forward with (input_ids, labels, attention_mask)
    loss, _ = lora_training((x, y, attention_mask))
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
    attention_mask = torch.randn(5, 10)

    loss, _ = lora_training((x, y, attention_mask))
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


def test_lora_trainer_train_with_various_batch_types():
    """Test LoraTrainer.train with batches of different types."""
    model = DummyLoRAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lora_trainer = LoraTrainer(model, optimizer=optimizer)

    # Mock the hybrid_model's __call__ method
    lora_trainer.hybrid_model = MagicMock(
        return_value=(torch.tensor(1.0, requires_grad=True), None)
    )

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

    class NonTensorDataset(Dataset):
        """Dataset with non-tensor items."""

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Test with dict batch
    dataset_dict = [{"input": torch.randn(5, 10), "label": torch.randn(5, 10)} for _ in range(2)]
    train_loader_dict: DataLoader = DataLoader(DictDataset(dataset_dict), batch_size=1)
    lora_trainer.train(train_loader_dict, num_epochs=1)

    # Test with list/tuple batch
    dataset_list = [(torch.randn(5, 10), torch.randn(5, 10)) for _ in range(2)]
    train_loader_list: DataLoader = DataLoader(ListDataset(dataset_list), batch_size=1)
    lora_trainer.train(train_loader_list, num_epochs=1)

    # Test with single tensor batch
    dataset_single = TensorDataset(torch.stack([torch.randn(5, 10) for _ in range(2)]))
    train_loader_single: DataLoader = DataLoader(dataset_single, batch_size=1)
    lora_trainer.train(train_loader_single, num_epochs=1)

    # Test with single non-tensor item batch
    dataset_non_tensor = NonTensorDataset(
        [42 for _ in range(2)]
    )  # Using integers as non-tensor data
    train_loader_non_tensor: DataLoader = DataLoader(dataset_non_tensor, batch_size=1)
    lora_trainer.train(train_loader_non_tensor, num_epochs=1)


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
