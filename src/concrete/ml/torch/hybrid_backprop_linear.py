"""Linear layer implementations for backprop FHE-compatible models."""

from torch import autograd, nn

# pylint: disable=arguments-differ,abstract-method


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
