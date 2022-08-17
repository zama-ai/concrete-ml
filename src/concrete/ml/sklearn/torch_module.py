"""Implement torch module."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class _GeneralizedLinearRegressionTorchModel(nn.Module):
    """A Torch module with one linear layer and a given activation function.

    This module is used for converting scikit-learn GLM models on Tweedie distributions (Normal,
    Poisson, Gamma and Inverse Gaussian distributions) into a Torch module without using the
    Hummingbird library.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        inverse_link: Callable,
        use_bias: bool = True,
    ):
        """Initialize the module.

        Args:
            input_size (int): Size of each input sample.
            output_size (int): Size of each output sample.
            inverse_link (Callable): Inverse link function used in the inference.
            use_bias (bool): If set to False, the linear layer will not learn a bias term.
                Default to True.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=use_bias)
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


class _LinearRegressionTorchModel(nn.Module):
    """A Torch module with only one custom linear layer.

    This module is used for applying the ReduceSum workaround to linear models.
    """

    def __init__(
        self,
        weights,
        bias=0.0,
    ):
        """Initialize the module.

        Args:
            weights (torch.tensor]): The weights learned by sklearn during to consider during the
                inference.
            bias (Optional[torch.tensor]): The bias terms learned by sklearn to consider during the
                inference. None is no bias has been considered. Default to None.
        """

        super().__init__()
        self.weights = weights
        self.bias = bias

    def forward(self, x: torch.Tensor):
        """Compute the inference y = X @ w + b.

        Args:
            x (torch.tensor): The input data.

        Returns:
            torch.Tensor: The predictions.
        """
        y_pred = x * self.weights
        y_pred = y_pred.sum(dim=1, keepdim=True)
        y_pred += self.bias
        return y_pred
