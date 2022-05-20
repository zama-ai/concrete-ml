"""Implement torch module."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class _LinearRegressionTorchModel(nn.Module):
    """A Torch module with one linear layer.

    This module is used for converting Scikit-learn GLM models on Tweedie distributions (Normal,
    Poisson, Gamma and Inverse Gaussian distributions) into a Torch module without using the
    Hummingbird library.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        inverse_link: Callable,
        bias: bool = True,
    ):
        """Initialize the module.

        Args:
            input_size (int): Size of each input sample.
            output_size (int): Size of each output sample.
            inverse_link (Callable): Inverse link function used in the inference.
            bias (bool): If set to False, the linear layer will not learn an additive bias.
                Default to True.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.inverse_link = inverse_link

    def forward(self, x: torch.Tensor):
        """Compute the inference.

        The computed expression is y = inverse_link(X @ w + b).

        Args:
            x (torch.tensor): The input data.

        Returns:
            torch.Tensor: The predictions.
        """
        y_pred = self.linear(x)
        y_pred = self.inverse_link(y_pred)
        return y_pred
