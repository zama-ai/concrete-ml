"""This module contains classes for LoRA (Low-Rank Adaptation) FHE training and custom layers."""

from typing import List, Tuple, Union

import torch
from torch import Tensor, autograd, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .hybrid_model import HybridFHEModel

try:
    from transformers import Conv1D as TransformerConv1D
except ImportError:  # pragma: no cover
    TransformerConv1D = None

# Create a tuple of linear layer classes to check against
LINEAR_LAYERS: tuple = (nn.Linear,)
if TransformerConv1D is not None:
    LINEAR_LAYERS = LINEAR_LAYERS + (TransformerConv1D,)


# pylint: disable=abstract-method
# pylint: disable=arguments-differ


class LoraTraining(torch.nn.Module):
    """LoraTraining module for fine-tuning with LoRA in a hybrid model setting.

    This class is designed to enable Low-Rank Adaptation (LoRA) fine-tuning
    in a hybrid model context. It allows selective execution of forward and
    backward passes in FHE.

    The class replaces standard linear layers with custom layers that are
    compatible with LoRA and FHE operations. It provides mechanisms to
    toggle between calibration and optimization modes.

    Args:
        model (torch.nn.Module): The base model with LoRA layers to be fine-tuned.
        n_layers_to_skip (int): Number of layers to skip. Linear layers that do not require
            gradient to be propagated are skipped. Defaults to 1.
        loss_fn (callable, optional): Loss function to compute the loss. If None, the model
            is expected to return a loss.
    """

    def __init__(self, model, n_layers_to_skip=1, loss_fn=None):
        super().__init__()

        # Assert that the model contains LoRA layers
        self.assert_has_lora_layers(model)

        self.inference_model = model
        self.replace_layers_with_custom(self.inference_model, n_layers_to_skip)

        self.calibrate = False
        self.loss_fn = loss_fn
        self.loss_scaling_factor = 1.0

    def set_loss_scaling_factor(self, loss_scaling_factor: float):
        """Set the loss scaling factor for gradient accumulation.

        Args:
            loss_scaling_factor (float): The factor to scale the loss by.
        """
        self.loss_scaling_factor = loss_scaling_factor

    @staticmethod
    def assert_has_lora_layers(model):
        """Assert that the model contains LoRA layers.

        Args:
            model (torch.nn.Module): The model to check for LoRA layers.

        Raises:
            ValueError: If the model does not contain any LoRA layers.
        """

        def is_lora_module(module):
            # Check for common LoRA attributes with case-insensitive matching
            lora_attributes = ["lora_a", "lora_b", "lora_dropout"]
            return any(
                hasattr(module, attr)
                or hasattr(module, attr.lower())
                or hasattr(module, attr.upper())
                for attr in lora_attributes
            )

        has_lora = any(is_lora_module(module) for module in model.modules())

        if not has_lora:
            raise ValueError("The model does not contain any detectable LoRA layers.")

        print("LoRA layers detected in the model.")

    @staticmethod
    def replace_layers_with_custom(model: nn.Module, n_layers_to_skip: int) -> None:
        """Replace linear layers with custom ones.

        Args:
            model (nn.Module): The model to replace layers in.
            n_layers_to_skip (int): Number of layers to skip.
        """

        def _replace(module: nn.Module):
            nonlocal n_layers_to_skip
            for name, child in list(module.named_children()):

                # Skip lora layers as they are computed on the client side
                if "lora" in name:
                    continue

                if isinstance(child, LINEAR_LAYERS):
                    if n_layers_to_skip > 0:
                        n_layers_to_skip -= 1

                        # Skip the first eligible layer
                        continue

                    # Determine if weights need to be transposed
                    weight_transposed = TransformerConv1D is not None and isinstance(
                        child, TransformerConv1D
                    )

                    # Create the CustomLinear layer
                    custom_layer = CustomLinear(
                        weight=child.weight,
                        bias=child.bias,
                        weight_transposed=weight_transposed,
                    )

                    # Replace the original layer with the custom layer
                    setattr(module, name, custom_layer)
                else:
                    # Recursively apply to child modules
                    _replace(child)

        _replace(model)

    def toggle_calibrate(self, enable: bool = True):
        """Toggle calibration mode.

        Args:
            enable (bool): Whether to enable calibration mode.
        """
        self.calibrate = enable

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, Union[Tensor, None]]:
        """Forward pass of the LoRA training module.

        Args:
            inputs (tuple): A tuple containing the input tensors.

        Returns:
            A tuple containing the original (unscaled) loss and None.

        Raises:
            ValueError: If the model does not return a loss and no loss function is provided.
        """
        assert (
            len(inputs) >= 2
        ), "Expected at least two inputs in the tuple: inputs (x) and targets (y)"

        # FIXME:
        # Remove when hybrid model supports multiple inputs modules
        # Unpack model inputs and labels
        *model_inputs, y = inputs

        if self.loss_fn is None:
            # Pass inputs and labels to the model
            outputs = self.inference_model(*model_inputs, labels=y)

            # Check if outputs is a dict and retrieve the loss
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
            else:
                loss = getattr(outputs, "loss", None)
            if loss is None:
                raise ValueError(
                    "The model did not return a loss.",
                    "Ensure that 'labels' are correctly provided or provide a loss_fn.",
                )
        else:
            # Forward pass without labels; compute loss manually
            outputs = self.inference_model(*model_inputs)
            if isinstance(outputs, dict) and "logits" in outputs:
                outputs = outputs["logits"]
            loss = self.loss_fn(outputs, y)

        # Scale the loss for gradient accumulation
        scaled_loss = loss / self.loss_scaling_factor

        # We need to set requires grad to the loss manually because the inference model's last
        # step is the "lm_head" layer, which might be detached from the graph by the hybrid model
        scaled_loss.requires_grad_(True)
        scaled_loss.backward()

        # Return the original (unscaled) loss for logging
        return loss.detach(), None


class LoraTrainer:
    """Trainer class for LoRA fine-tuning with FHE support.

    This class handles the training loop, optimizer, scheduler,
    and integrates with the hybrid model.

    Args:
        model (nn.Module): The base model with LoRA layers to be fine-tuned.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (callable): Loss function to compute the loss.
        lr_scheduler (optional): Learning rate scheduler.
        training_args (dict): Training arguments.
        n_layers_to_skip (int): Number of layers to skip. Defaults to 1.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        lr_scheduler=None,
        training_args=None,
        n_layers_to_skip=1,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_args = training_args or {}
        self.gradient_accumulation_steps = self.training_args.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = self.training_args.get("max_grad_norm", None)

        # Create the LoRA training module
        self.lora_training_module = LoraTraining(
            model, n_layers_to_skip=n_layers_to_skip, loss_fn=loss_fn
        )

        # Determine modules to be executed remotely
        remote_names = get_remote_names(self.lora_training_module)

        # Create the hybrid model
        self.hybrid_model = HybridFHEModel(self.lora_training_module, module_names=remote_names)

    def compile(self, inputset, n_bits=8):
        """Compile the hybrid model with the given input set.

        Args:
            inputset (tuple): Input set for compilation.
            n_bits (int): Bit width for quantization.
        """
        self.lora_training_module.toggle_calibrate(enable=True)
        self.hybrid_model.compile_model(inputset, n_bits=n_bits)
        self.lora_training_module.toggle_calibrate(enable=False)

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        fhe: str = "simulate",
    ):
        """Train the model using the hybrid FHE model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            fhe (str): FHE mode ('disable', 'simulate', 'execute' or 'torch').
        """
        device = torch.device("cpu")
        self.lora_training_module.to(device)
        self.lora_training_module.inference_model.train()

        # Set the loss scaling factor for gradient accumulation
        self.lora_training_module.set_loss_scaling_factor(self.gradient_accumulation_steps)

        epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            total_loss = 0.0
            self.optimizer.zero_grad()  # Zero gradients at the start of the epoch

            for step, batch in enumerate(train_loader):

                # Convert batch to tuple and move to device
                if isinstance(batch, dict):

                    # Convert dict to tuple of values and move tensors to device
                    batch = tuple(
                        v.to(device) if isinstance(v, torch.Tensor) else v for v in batch.values()
                    )
                elif isinstance(batch, (tuple, list)):

                    # Move tensor items to the device
                    batch = tuple(
                        item.to(device) if isinstance(item, torch.Tensor) else item
                        for item in batch
                    )
                else:

                    # If it's a single item, wrap it in a tuple and move to device if it's a tensor
                    batch = (batch.to(device) if isinstance(batch, torch.Tensor) else batch,)

                # Forward pass
                loss, _ = self.hybrid_model(batch, fhe=fhe)

                # Loss scaling and backward is done inside LoraTraining

                # Accumulate loss for logging
                total_loss += loss.item()

                # Update weights and reset gradients after specified steps
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(
                    train_loader
                ):
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.lora_training_module.parameters(), self.max_grad_norm
                        )

                    # Optimizer step
                    self.optimizer.step()

                    # Scheduler step
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # Zero gradients
                    self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            epoch_pbar.set_postfix(
                {
                    "Epoch": epoch,
                    "Avg Loss": f"{avg_loss:.4f}",
                    "FHE Mode": fhe,
                }
            )

        print(f"Training completed. Final Avg Loss: {avg_loss:.4f}, FHE Mode: {fhe}")

    def save_and_clear_private_info(self, path):
        """Save the model and remove private information.

        Args:
            path (str): The path to save the model.
        """
        self.hybrid_model.save_and_clear_private_info(path)


class ForwardModuleLinear(nn.Module):
    """Forward module for linear layers."""

    def __init__(self, weight, bias=None, weight_transposed=False):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.weight_transposed = weight_transposed  # If True, weight is (in_features, out_features)

    def forward(self, input_tensor):
        """Forward pass for linear layers.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor after applying the linear transformation.
        """
        if self.weight_transposed:
            # Weight is (in_features, out_features)
            output = input_tensor @ self.weight
        else:
            # Weight is (out_features, in_features)
            output = input_tensor @ self.weight.t()
        if self.bias is not None:
            output += self.bias
        return output


class BackwardModuleLinear(nn.Module):
    """Backward module for linear layers."""

    def __init__(self, weight, weight_transposed=False):
        super().__init__()
        self.weight = weight
        self.weight_transposed = weight_transposed

    def forward(self, grad_output):
        """Backward pass for linear layers.

        Args:
            grad_output: The gradient output tensor.

        Returns:
            The gradient input tensor after applying the backward pass.
        """
        if self.weight_transposed:
            grad_input = grad_output @ self.weight.t()
        else:
            grad_input = grad_output @ self.weight
        return grad_input


class CustomLinear(nn.Module):
    """Custom linear module."""

    def __init__(self, weight, bias=None, weight_transposed=False):
        super().__init__()
        self.forward_module = ForwardModuleLinear(weight, bias, weight_transposed)
        self.backward_module = BackwardModuleLinear(weight, weight_transposed)

    def forward(self, input_tensor):
        """Forward pass of the custom linear module.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor after applying the custom linear module.
        """
        return ForwardBackwardModule.apply(input_tensor, self.forward_module, self.backward_module)


class ForwardBackwardModule(autograd.Function):
    """Custom autograd function for forward and backward passes."""

    @staticmethod
    def forward(ctx, input_tensor, forward_module, backward_module):
        """Forward pass of the custom autograd function.

        Args:
            ctx: The context object.
            input_tensor: The input tensor.
            forward_module: The forward module.
            backward_module: The backward module.

        Returns:
            The output tensor after applying the forward pass.
        """
        ctx.backward_module = backward_module
        output = forward_module.forward(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the custom autograd function.

        Args:
            ctx: The context object.
            grad_output: The gradient output tensor.

        Returns:
            The gradient input tensor after applying the backward pass.
        """
        backward_module = ctx.backward_module
        grad_input = backward_module.forward(grad_output)

        # grad_weight and grad_bias are not needed when computing the backward for LoRA
        return grad_input, None, None


def get_remote_names(model: nn.Module, include_embedding_layers: bool = False) -> List[str]:
    """Get names of modules to be executed remotely.

    Args:
        model (nn.Module): The model to inspect.
        include_embedding_layers (bool): Whether to include embedding layers.

    Returns:
        List[str]: List of module names to be executed remotely.
    """
    remote_names = []
    for name, module in model.named_modules():
        # Skip if the name contains 'lora' since they will be done on client side
        if "lora" in name:
            continue

        # Skip 'lm_head' if embedding layers are not included
        is_lm_head = "lm_head" in name
        if is_lm_head and not include_embedding_layers:
            continue

        # Handle different module types
        if isinstance(module, LINEAR_LAYERS):
            remote_names.append(name)
        elif isinstance(module, CustomLinear):
            remote_names.append(f"{name}.forward_module")
            remote_names.append(f"{name}.backward_module")
        elif include_embedding_layers and (isinstance(module, nn.Embedding) or is_lm_head):
            remote_names.append(name)

    return remote_names
