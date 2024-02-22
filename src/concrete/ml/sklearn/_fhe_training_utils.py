"""Utility functions for FHE training."""

from typing import Tuple

import numpy
import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def binary_cross_entropy(y_true: numpy.ndarray, logits: numpy.ndarray):
    """Binary cross-entropy with logits.

    Arguments:
        y_true (numpy.ndarray): The ground truth values.
        logits (numpy.ndarray): The computed logits.

    Returns:
        The binary cross entropy loss value.
    """
    return binary_cross_entropy_with_logits(torch.Tensor(logits), torch.Tensor(y_true)).item()


class LogisticRegressionTraining(torch.nn.Module):
    """Logistic Regression training module.

    We use this torch module to represent the training of a model in order to be able to compile it
    to FHE.

    The forward function iterates the SGD over a given certain number of times.
    """

    def __init__(self, iterations: int = 1, learning_rate: float = 1.0, fit_bias: bool = True):
        """Instantiate the model.

        Args:
            iterations (int): The number of times over which to iterate the SGD during the forward.
            learning_rate (float): The learning rate.
            fit_bias (bool): If the bias will be taken into account or not.
        """
        super().__init__()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.fit_bias = fit_bias

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SGD forward.

        This function iterates the SGD over a given certain number of times and returns the new
        weights and bias.

        Args:
            features (torch.Tensor): The input values.
            targets (torch.Tensor): THe target values.
            weights (torch.Tensor): The initial weight values.
            bias (torch.Tensor): The initial bias values.

        Returns:
            torch.Tensor: The updated weight values.
            torch.Tensor: The updated bias values.
        """
        # Weights: (1, features, n_targets)
        # Bias: (1, n_targets, 1)
        for _ in range(self.iterations):

            # Compute the inference
            if self.fit_bias:
                logits = features @ weights + bias  # (n_samples, n_targets, 1)
            else:
                logits = features @ weights  # (n_samples, n_targets, 1)
            probits = torch.sigmoid(logits)  # (n_samples, n_targets, 1)

            # Compute the gradients
            derive_z = probits - targets  # (n_samples, n_targets, 1)
            derive_weights = (
                features.transpose(1, 2) @ derive_z / features.size(1)
            )  # (1, n_features, 1)

            derive_bias = derive_z.sum(dim=1, keepdim=True) / derive_z.size(1)  # (1, n_targets, 1)

            # Update the weight and bias values
            weights -= self.learning_rate * derive_weights
            if self.fit_bias:
                bias -= self.learning_rate * derive_bias
            else:
                bias = bias * torch.zeros(bias.shape)

        # Should we clip the parameters to the min-max values?
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4206

        # (1, n_features, n_targets), (1, n_targets, 1)
        return weights, bias
