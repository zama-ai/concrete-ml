"""This module contains classes for LoRA (Low-Rank Adaptation) training and custom layers."""

from typing import List

import torch

try:
    from transformers import Conv1D as TransformerConv1D
except ImportError:
    TransformerConv1D = None

# Create a tuple of linear layer classes to check against
LINEAR_LAYERS: tuple = (torch.nn.Linear,)
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
        inference_model (torch.nn.Module): The base model to be fine-tuned.

    """

    def __init__(self, inference_model) -> None:
        super().__init__()

        self.inference_model = inference_model

        self.replace_layers_with_custom(self.inference_model)

        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = None
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = None

        self.calibrate = False
        self.run_optimizer = False

    @staticmethod
    def replace_layers_with_custom(model: torch.nn.Module, skip_first: bool = True):
        """Replace linear layers with custom ones.

        This method replaces eligible linear layers in the model with custom layers
        that are compatible with the LoRA training procedure.

        Args:
            model (torch.nn.Module): The model to replace layers in.
            skip_first (bool): Whether to skip the first eligible layer.
        """
        # Flag to track if the first layer has been skipped
        skipped = False

        def _replace(module: torch.nn.Module):
            nonlocal skipped
            for name, child in list(module.named_children()):
                # Skip modules containing "lora" in their name
                if "lora" in name:
                    continue

                if isinstance(child, LINEAR_LAYERS):
                    if skip_first and not skipped:
                        skipped = True

                        # Skip the first eligible layer
                        continue

                    # Determine if weights need to be transposed
                    weight_transposed = TransformerConv1D is not None and isinstance(
                        child, TransformerConv1D
                    )

                    # Create the CustomLinear layer
                    custom_layer = CustomLinear(
                        weight=child.weight, bias=child.bias, weight_transposed=weight_transposed
                    )

                    # Replace the original layer with the custom layer
                    setattr(module, name, custom_layer)
                else:
                    # Recursively apply to child modules
                    _replace(child)

        _replace(model)

    def update_training_parameters(
        self, optimizer=None, lr_scheduler=None, loss_fn=None, training_args=None
    ):
        """Update training parameters for the LoRA module.

        Args:
            optimizer (optional): The optimizer to use for training.
            lr_scheduler (optional): The learning rate scheduler to use for training.
            loss_fn (callable, optional): Loss function to compute the loss.
            training_args (dict or namespace, optional): Training arguments containing
                'gradient_accumulation_steps' and 'max_grad_norm'.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

        if training_args is not None:
            # Check if training_args is a dict or an object with attributes
            if isinstance(training_args, dict):
                self.gradient_accumulation_steps = training_args.get(
                    "gradient_accumulation_steps", 1
                )
                self.max_grad_norm = training_args.get("max_grad_norm", None)
            else:
                self.gradient_accumulation_steps = getattr(
                    training_args, "gradient_accumulation_steps", 1
                )
                self.max_grad_norm = getattr(training_args, "max_grad_norm", None)
        else:
            self.gradient_accumulation_steps = 1
            self.max_grad_norm = None

    def forward(self, inputs):
        """Forward pass of the LoRA training module.

        Args:
            inputs: A tuple containing input tensors and labels.

        Returns:
            A tuple containing the loss and gradient norm.

        Raises:
            ValueError: If the model does not return a loss when `self.loss_fn` is None.
        """
        # Remove this once hybrid model supports multiple inputs
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4568
        x, y = inputs

        # Forward pass
        if self.loss_fn is None:

            # Assume model computes loss internally
            outputs = self.inference_model(x, labels=y)

            # Use getattr to safely access the loss attribute
            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise ValueError(
                    "The model did not return a loss. Ensure that 'labels' are correctly provided."
                )
        else:
            outputs = self.inference_model(x)
            loss = self.loss_fn(outputs, y)

        loss = loss / self.gradient_accumulation_steps

        # Update gradients
        # We need to set requires grad to the loss manually because the inference model's last
        # step is the "lm_head" layer, which might be detached from the graph by the hybrid model
        loss.requires_grad_(True)
        loss.backward()

        grad_norm = None
        if not self.calibrate and self.run_optimizer:
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.inference_model.parameters(), max_norm=self.max_grad_norm, norm_type=2
                )

            if self.optimizer is not None:
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.inference_model.zero_grad()

        # Clean gradients after calibration
        elif self.calibrate:
            self.inference_model.zero_grad()

        return (loss, grad_norm)

    def toggle_calibrate(self, enable: bool = True):
        """Toggle calibration mode.

        Args:
            enable (bool): Whether to enable calibration mode.
        """
        self.calibrate = enable

    def toggle_run_optimizer(self, enable: bool = True):
        """Toggle optimizer execution.

        Args:
            enable (bool): Whether to enable optimizer execution.
        """
        self.run_optimizer = enable


class ForwardModuleLinear(torch.nn.Module):
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


class BackwardModuleLinear(torch.nn.Module):
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


class CustomLinear(torch.nn.Module):
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


class ForwardBackwardModule(torch.autograd.Function):
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


def get_remote_names(model: torch.nn.Module, include_embedding_layers: bool = False) -> List[str]:
    """Get names of modules to be executed remotely.

    Args:
        model (torch.nn.Module): The model to inspect.
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
        elif include_embedding_layers and (isinstance(module, torch.nn.Embedding) or is_lm_head):
            remote_names.append(name)

    return remote_names
