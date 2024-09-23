"""This module contains classes for LoRA (Low-Rank Adaptation) training and custom layers."""

import torch

# pylint: disable=abstract-method
# pylint: disable=arguments-differ


class LoraTraining(torch.nn.Module):
    """LoraTraining module for fine-tuning with LoRA."""

    SUPPORTED_MODELS = ["gpt2"]

    def __init__(self, inference_model, gradient_accumulation_steps) -> None:
        super().__init__()

        self.inference_model = inference_model

        # Validate the base model type
        self._validate_model_type()

        self.optimizer = None
        self.lr_scheduler = None

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = None

        self.calibrate = False
        self.run_optimizer = False

    def _validate_model_type(self):
        """Validate the model type.

        Raises:
            ValueError: If the model type is not supported.
        """
        try:
            # Access the base model from PeftModelForCausalLM
            base_model = self.inference_model.base_model.model

            # Retrieve the model type from the configuration
            model_type = getattr(base_model.config, "model_type", None)

            if model_type not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model type: '{model_type}'. "
                    f"Supported models are: {self.SUPPORTED_MODELS}"
                )

        except AttributeError as e:
            raise ValueError(
                "Unable to determine the base model type. "
                "Ensure that the inference_model has a "
                "'base_model.model.config.model_type' attribute."
            ) from e

    def update_training_parameters(self, optimizer, lr_scheduler, training_args):
        """Update training parameters for the LoRA module.

        Args:
            optimizer: The optimizer to use for training.
            lr_scheduler: The learning rate scheduler to use for training.
            training_args: The training arguments containing gradient
                accumulation steps and max grad norm.
        """
        assert self.gradient_accumulation_steps == training_args.gradient_accumulation_steps

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = training_args.max_grad_norm

    def forward(self, inputs):
        """Forward pass of the LoRA training module.

        Args:
            inputs: A tuple containing input tensors and labels.

        Returns:
            A tuple containing the loss and gradient norm.

        Raises:
            ValueError: If the model does not return a loss.
        """
        # Remove this once hybrid model supports multiple inputs
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4568
        x, y = inputs

        # Correctly pass labels as a keyword argument
        outputs = self.inference_model(x, labels=y)

        # Use getattr to safely access the loss attribute
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError(
                "The model did not return a loss. Ensure that 'labels' are correctly provided."
            )

        loss = loss / self.gradient_accumulation_steps

        # Update gradients
        # We need to set requires grad to the loss manually because the inference model's last
        # step is the "lm_head" layer, which is detached from the graph by the hybrid model
        loss.requires_grad_(True)
        loss.backward()

        grad_norm = None
        if not self.calibrate and self.run_optimizer:
            assert self.optimizer is not None
            assert self.lr_scheduler is not None
            assert self.max_grad_norm is not None

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.inference_model.parameters(), max_norm=self.max_grad_norm, norm_type=2
            )

            self.optimizer.step()
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


class ForwardModule(torch.nn.Module):
    """Forward module for custom convolution."""

    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = weight  # Assume weight is passed as a pre-initialized tensor
        self.bias = bias

    def forward(self, input_tensor):
        """Forward pass of the forward module.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor after applying the forward pass.
        """
        output = input_tensor @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class BackwardModule(torch.nn.Module):
    """Backward module for custom convolution."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight  # This is the same weight used in ForwardModule

    def forward(self, grad_output):
        """Forward pass of the backward module.

        Args:
            grad_output: The gradient output tensor.

        Returns:
            The gradient input tensor after applying the backward pass.
        """
        return grad_output @ self.weight.t()


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

        # grad_weight and grad_bias are not needed when computing the backward for lora
        return grad_input, None, None


class CustomConv1D(torch.nn.Module):
    """Custom 1D convolution module."""

    def __init__(self, weight, bias=None):
        super().__init__()
        self.forward_module = ForwardModule(weight, bias=bias)
        self.backward_module = BackwardModule(weight)

    def forward(self, input_tensor):
        """Forward pass of the custom 1D convolution.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor after applying the custom 1D convolution.
        """
        return ForwardBackwardModule.apply(input_tensor, self.forward_module, self.backward_module)


class CustomLinear(torch.nn.Module):
    """Custom linear module."""

    def __init__(self, weight, bias=None):
        super().__init__()
        self.forward_module = ForwardModule(weight, bias=bias)
        self.backward_module = BackwardModule(weight)

    def forward(self, input_tensor):
        """Forward pass of the custom linear module.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor after applying the custom linear module.
        """
        return ForwardBackwardModule.apply(input_tensor, self.forward_module, self.backward_module)
