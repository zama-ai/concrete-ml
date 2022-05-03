"""Implement torch module."""
from __future__ import annotations

import torch
from sklearn.linear_model._glm.link import IdentityLink, LogLink
from torch import nn

from ..common.debugging.custom_assert import assert_true

LINK_NAMES = {LogLink: "log", IdentityLink: "identity"}


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
        bias: bool = True,
        link: str = "identity",
    ):
        """Initialize the module.

        Args:
            input_size (int): Size of each input sample.
            output_size (int): Size of each output sample.
            bias (bool): If set to False, the linear layer will not learn an additive bias.
                Default to True.
            link (str): Link function used in the inference, can either be "identity" or "log".
                Default to 'identity'.
        """
        # Making sure we handle the extracted link function
        assert_true(
            link in LINK_NAMES.values(), f"Link must be of either {', '.join(LINK_NAMES.values())}."
        )

        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.link = link

    def forward(self, x: torch.Tensor):
        """Compute the inference.

        If the link function is 'identity', a simple linear inference is computed. If 'log' was
        chosen, an exponential function is applied to the outputs and the inference expression
        becomes y = exp(X @ w + b).

        Args:
            x (torch.tensor): The input data.

        Returns:
            torch.Tensor: The predictions.
        """
        y_pred = self.linear(x)

        if self.link == "log":
            y_pred = torch.exp(y_pred)

        return y_pred
