"""Implement torch module."""
from __future__ import annotations

from typing import Optional

import numpy
import torch
from torch import nn


class _LinearTorchModel(nn.Module):
    """A Torch module with one linear layer."""

    def __init__(
        self,
        weight: numpy.ndarray,
        bias: Optional[numpy.ndarray] = None,
    ):
        """Initialize the module using some pre-trained weight and bias values.

        Args:
            weight (numpy.ndarray): The weight values.
            bias (Optional[numpy.ndarray]): The bias values. If None, no bias term will be
                considered. Default to None.
        """
        super().__init__()

        # Extract the input and output sizes
        input_size = weight.shape[0]
        output_size = weight.shape[1] if len(weight.shape) > 1 else 1

        use_bias = bias is not None
        self.linear = nn.Linear(input_size, output_size, bias=use_bias)

        # Update the module's weights and bias
        self.linear.weight.data = torch.from_numpy(weight).reshape(output_size, input_size)

        if use_bias:
            self.linear.bias.data = torch.tensor(bias)

    def forward(self, x: torch.Tensor):
        """Compute a linear inference.

        Args:
            x (torch.tensor): The input data.

        Returns:
            torch.Tensor: The predictions.
        """
        return self.linear(x)
