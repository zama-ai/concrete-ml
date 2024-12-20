"""This module contains classes for LoRA (Low-Rank Adaptation) FHE training and custom layers."""

from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .hybrid_backprop_linear import CustomLinear
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


def try_dict(obj):
    """Try to convert the object to a dict.

    Args:
        obj: The object to convert to a dict.

    Returns:
        The object converted to a dict or None if the conversion fails.
    """
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return None


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
        n_layers_to_skip_for_backprop (int): Number of initial linear layers to keep as standard
            layers. Since the first layer doesn't need backpropagation (no previous layer to
            update), we typically skip 1 layer. Defaults to 1.
        loss_fn (callable, optional): Loss function to compute the loss. If None, the model
            is expected to return a loss.
    """

    def __init__(self, model, n_layers_to_skip_for_backprop=1, loss_fn=None):
        super().__init__()

        # Check if model accepts labels when no loss_fn is provided
        if loss_fn is None:
            from inspect import signature

            forward_sig = signature(model.forward)
            if "labels" not in forward_sig.parameters:
                raise ValueError(
                    "When no loss_fn is provided, the model's forward method"
                    "must accept a 'labels' parameter"
                )

        # Assert that the model contains LoRA layers
        self.assert_has_lora_layers(model)

        self.inference_model = model
        self.replace_layers_with_custom(self.inference_model, n_layers_to_skip_for_backprop)

        self.calibrate = False
        self.loss_fn = loss_fn
        self.loss_scaling_factor = 1.0

    def set_loss_scaling_factor(self, loss_scaling_factor: float):
        """Set a scaling factor for the loss to account for gradient accumulation.

        This ensures that gradients are correctly averaged over multiple
        mini-batches when performing gradient accumulation, preventing them
        from being scaled up by the number of accumulation steps.

        Args:
            loss_scaling_factor (float): The number of gradient accumulation steps.
                                        The loss will be divided by this factor
                                        before backpropagation.
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
    def replace_layers_with_custom(model: nn.Module, n_layers_to_skip_for_backprop: int) -> None:
        """Replace linear layers with custom ones.

        Args:
            model (nn.Module): The model to replace layers in.
            n_layers_to_skip_for_backprop (int): Number of initial linear layers to keep as standard
                layers. Since the first layer doesn't need backpropagation (no previous layer to
                update), we typically skip 1 layer.
        """

        def _replace(module: nn.Module):
            nonlocal n_layers_to_skip_for_backprop
            for name, child in list(module.named_children()):

                # Skip lora layers as they are computed on the client side
                if "lora" in name:
                    continue

                if isinstance(child, LINEAR_LAYERS):
                    if n_layers_to_skip_for_backprop > 0:
                        n_layers_to_skip_for_backprop -= 1

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
            len(inputs) >= 2 and len(inputs) <= 3
        ), "Expected at least two inputs in the tuple: inputs (x) and targets (y)"

        # Unpack depending on how many inputs we have
        if len(inputs) == 2:
            input_ids, labels = inputs
            attention_mask = None
        else:
            input_ids, labels, attention_mask = inputs

            # Validate attention mask
            assert torch.all(
                torch.logical_or(attention_mask == 0, attention_mask == 1)
            ), "Invalid attention mask provided. Attention mask should only contain 0s and 1s."

        if self.loss_fn is None:
            # Pass inputs and labels to the model
            if attention_mask is not None:
                outputs = self.inference_model(
                    input_ids, labels=labels, attention_mask=attention_mask
                )
            else:
                outputs = self.inference_model(input_ids, labels=labels)

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
            if attention_mask is not None:
                logits = self.inference_model(input_ids, attention_mask=attention_mask)
            else:
                logits = self.inference_model(input_ids)

            # If logits is a dict with 'logits' key, extract it
            if isinstance(logits, dict) and "logits" in logits:
                logits = logits["logits"]

            loss = self.loss_fn(logits, labels)

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
        n_layers_to_skip_for_backprop (int): Number of initial linear layers to keep as standard
            layers. Since the first layer doesn't need backpropagation (no previous layer to
            update), we typically skip 1 layer. Defaults to 1.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        lr_scheduler=None,
        training_args=None,
        n_layers_to_skip_for_backprop=1,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_args = training_args or {}
        self.gradient_accumulation_steps = self.training_args.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = self.training_args.get("max_grad_norm", None)

        # Create the LoraTraining module
        self.lora_training_module = LoraTraining(
            model, n_layers_to_skip_for_backprop=n_layers_to_skip_for_backprop, loss_fn=loss_fn
        )

        # Determine modules to be executed remotely
        self.remote_names = get_remote_names(self.lora_training_module)

        # Create the hybrid model
        self.hybrid_model = HybridFHEModel(
            self.lora_training_module, module_names=self.remote_names
        )

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
        device = torch.device("cuda")
        #        self.lora_training_module.to(device)
        self.lora_training_module.inference_model.train()

        # Set the loss scaling factor for gradient accumulation
        self.lora_training_module.set_loss_scaling_factor(self.gradient_accumulation_steps)

        epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            total_loss = 0.0
            self.optimizer.zero_grad()  # Zero gradients at the start of the epoch

            for step, batch in enumerate(train_loader):

                # Convert the batch to a tuple of inputs on the device.
                if batch_dict := try_dict(batch):
                    batch = batch_dict
                    # Convert dict to tuple of values and move them to the device
                    batch = tuple(
                        v.to(device) if isinstance(v, torch.Tensor) else v for v in batch.values()
                    )
                elif isinstance(batch, (tuple, list)):
                    # Move tuple/list elements to the device
                    batch = tuple(
                        item.to(device) if isinstance(item, torch.Tensor) else item
                        for item in batch
                    )
                else:
                    # If it is a single non-tensor item, wrap it in a tuple
                    batch = (batch,)

                # Forward pass through the hybrid model
                loss, _ = self.hybrid_model(batch, fhe=fhe)

                # Loss scaling and backward is done inside LoraTraining

                # Accumulate loss for logging
                total_loss += loss.item()

                # Update weights after gradient accumulation steps
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
