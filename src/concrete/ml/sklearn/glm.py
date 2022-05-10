"""Implement sklearn's Generalized Linear Models (GLM)."""
from __future__ import annotations

import copy
from typing import Tuple, Union

import numpy
import sklearn
import torch
from sklearn._loss.glm_distribution import ExponentialDispersionModel, TweedieDistribution
from sklearn.linear_model._glm.link import BaseLink

from ..common.debugging.custom_assert import assert_true
from ..quantization import PostTrainingAffineQuantization
from ..torch.numpy_module import NumpyModule
from .base import SklearnLinearModelMixin
from .torch_module import LINK_NAMES, _LinearRegressionTorchModel


# pylint: disable=too-many-instance-attributes
class _GeneralizedLinearRegressor(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """Regression via a penalized Generalized Linear Model (GLM) with FHE."""

    # The inheritance method does not inherit directly from the related Sklearn model and therefore
    # is not initialized by using super()
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 2,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, TweedieDistribution] = "auto",
        solver: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        self.n_bits = n_bits
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    # pylint: enable=super-init-not-called

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs) -> None:
        """Fit the GLM regression quantized model.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.
        """

        # Copy X
        X = copy.deepcopy(X)

        # Retrieving the Sklearn parameters
        params = self.get_params()
        params.pop("n_bits", None)

        # Initialize a Sklearn generalized linear regressor model
        # pylint: disable=attribute-defined-outside-init
        self.sklearn_model = self.sklearn_alg(**params)

        # Train
        self.sklearn_model.fit(X, y, *args, **kwargs)

        # Extract the weights
        weight = self.sklearn_model.coef_

        # Store the weights in an attribute used for .predict()
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/445
        # Improve the computation of number of outputs when refactoring is done, remove self.coef_
        self.coef_ = self.sklearn_model.coef_

        # Extract the input and output sizes
        input_size = weight.shape[0]
        output_size = weight.shape[1] if len(weight.shape) > 1 else 1

        # pylint: disable=protected-access

        link = self.sklearn_model._link_instance.__class__

        # Making sure we handle the extracted link function
        assert_true(link in LINK_NAMES, f"Link must be a member of {LINK_NAMES.keys()}.")

        # Initialize the Torch model. Using a small Torch model that reproduces the proper
        # inference is necessary in this case because the Hummingbird library, which is used for
        # converting a Sklearn model into an ONNX one, doesn't not support GLMs. Also, the Torch
        # module can be given to the NumpyModule class the same way it is done for its ONNX
        # equivalent, thus making the initial workflow still relevant.
        torch_model = _LinearRegressionTorchModel(
            input_size=input_size,
            output_size=output_size,
            bias=self.fit_intercept,
            link=LINK_NAMES[link],
        )

        # pylint: enable=protected-access

        # Update the Torch model's weights and bias using the Sklearn model's one
        torch_model.linear.weight.data = torch.from_numpy(weight).reshape(output_size, input_size)
        if self.fit_intercept:
            torch_model.linear.bias.data = torch.tensor(self.sklearn_model.intercept_)

        # Create a NumpyModule from the Torch model
        numpy_module = NumpyModule(torch_model, dummy_input=torch.from_numpy(X[0]))

        # Apply post-training quantization
        post_training = PostTrainingAffineQuantization(
            n_bits=self.n_bits, numpy_model=numpy_module, is_signed=True
        )

        # Calibrate and create quantize module
        self.quantized_module = post_training.quantize_module(X)

        # pylint: enable=attribute-defined-outside-init

    def fit_benchmark(
        self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs
    ) -> Tuple[_GeneralizedLinearRegressor, sklearn.linear_model._glm.GeneralizedLinearRegressor]:
        """Fit the sklearn and quantized models.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.

        Returns:
            Tuple[_GeneralizedLinearRegressor,
                sklearn.linear_model._glm.GeneralizedLinearRegressor]:
                The quantized and sklearn GLM regressor.
        """
        # Train the sklearn model without X quantized
        sklearn_model = self.sklearn_alg(*args, **kwargs)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the quantized model
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model


# pylint: enable=too-many-instance-attributes


class PoissonRegressor(_GeneralizedLinearRegressor):
    """A Poisson regression model with FHE."""

    sklearn_alg = sklearn.linear_model.PoissonRegressor

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 2,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            n_bits=n_bits,
            alpha=alpha,
            fit_intercept=fit_intercept,
            family="poisson",
            link="log",
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )


class GammaRegressor(_GeneralizedLinearRegressor):
    """A Gamma regression model with FHE."""

    sklearn_alg = sklearn.linear_model.GammaRegressor

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 2,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            n_bits=n_bits,
            alpha=alpha,
            fit_intercept=fit_intercept,
            family="gamma",
            link="log",
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )


class TweedieRegressor(_GeneralizedLinearRegressor):
    """A Tweedie regression model with FHE."""

    sklearn_alg = sklearn.linear_model.TweedieRegressor

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 2,
        power: float = 0.0,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        link: Union[str, BaseLink] = "auto",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            n_bits=n_bits,
            alpha=alpha,
            fit_intercept=fit_intercept,
            family=TweedieDistribution(power=power),
            link=link,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )
        self.power = power
