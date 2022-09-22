"""Implement sklearn's Generalized Linear Models (GLM)."""
from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Union

import numpy
import sklearn
import torch

from ..common.check_inputs import check_X_y_and_assert
from ..common.debugging.custom_assert import assert_true
from ..quantization import PostTrainingAffineQuantization
from ..torch.numpy_module import NumpyModule
from .base import SklearnLinearModelMixin
from .torch_module import _LinearRegressionTorchModel


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
        solver: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        self.n_bits = n_bits
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)

    # pylint: enable=super-init-not-called

    @property
    def onnx_model(self):
        return self._onnx_model_

    def fit(self, X, y: numpy.ndarray, *args, **kwargs) -> None:
        """Fit the GLM regression quantized model.

        Args:
            X : The training data, which can be:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): The target data.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.
        """
        # GLMS don't handle the use of a sum workaround
        kwargs.pop("use_sum_workaround", None)

        # Copy X and check that is has a proper type
        X = copy.deepcopy(X)
        X, y = check_X_y_and_assert(X, y)

        # Retrieving the Sklearn parameters
        params = self.get_params()
        params.pop("n_bits", None)

        # Initialize a sklearn generalized linear model
        # pylint: disable=attribute-defined-outside-init
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(X, y, *args, **kwargs)

        # Extract the weights
        weight = self.sklearn_model.coef_

        # Extract the input and output sizes
        input_size = weight.shape[0]
        output_size = weight.shape[1] if len(weight.shape) > 1 else 1

        # Initialize the Torch model. Using a small Torch model that reproduces the proper
        # inference is necessary in this case because the Hummingbird library, which is used for
        # converting a Sklearn model into an ONNX one, doesn't not support GLMs. Also, the Torch
        # module can be given to the NumpyModule class the same way it is done for its ONNX
        # equivalent, thus making the initial workflow still relevant.
        torch_model = _LinearRegressionTorchModel(
            input_size=input_size,
            output_size=output_size,
            use_bias=self.fit_intercept,
        )

        # Update the Torch model's weights and bias using the Sklearn model's one
        torch_model.linear.weight.data = torch.from_numpy(weight).reshape(output_size, input_size)
        if self.fit_intercept:
            torch_model.linear.bias.data = torch.tensor(self.sklearn_model.intercept_)

        # Create a NumpyModule from the Torch model
        numpy_module = NumpyModule(
            torch_model,
            dummy_input=torch.from_numpy(X[0]),
        )
        self._onnx_model_ = numpy_module.onnx_model

        # Apply post-training quantization
        post_training = PostTrainingAffineQuantization(
            n_bits=self.n_bits, numpy_model=numpy_module, is_signed=True
        )

        # Calibrate and create quantize module
        self.quantized_module_ = post_training.quantize_module(X)

        # pylint: enable=attribute-defined-outside-init

    def post_processing(
        self, y_preds: numpy.ndarray, already_dequantized: bool = False
    ) -> numpy.ndarray:
        """Post-processing the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions to post-process.
            already_dequantized (bool): Wether the inputs were already dequantized or not. Default
                to False.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        # If y_preds were already dequantized previously, there is no need to do so once again.
        # This step is necessary for the client-server workflow as the post_processing method
        # is directly called on the quantized outputs, contrary to the base class' predict method.
        if not already_dequantized:
            y_preds = self.quantized_module_.dequantize_output(y_preds)

        return self._inverse_link(y_preds)

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict on user data.

        Predict on user data using either the quantized clear model, implemented with tensors, or,
        if execute_in_fhe is set, using the compiled FHE circuit.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute the inference in FHE. Default to False.

        Returns:
            numpy.ndarray: The model's predictions.
        """
        y_preds = super().predict(X, execute_in_fhe=execute_in_fhe)
        y_preds = self.post_processing(y_preds, already_dequantized=True)
        return y_preds

    @abstractmethod
    def _inverse_link(self, y_preds):
        """Apply the link function's inverse on the inputs.

        Args:
            y_preds (numpy.ndarray): The input data.
        """


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
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _inverse_link(self, y_preds) -> numpy.ndarray:
        """Apply the link function's inverse on the inputs.

        PoissonRegressor uses the exponential function.

        Args:
            y_preds (numpy.ndarray): The input data.

        Returns:
            The model's final predictions.
        """
        return numpy.exp(y_preds)


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
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _inverse_link(self, y_preds) -> numpy.ndarray:
        """Apply the link function's inverse on the inputs.

        GammaRegressor uses the exponential function.

        Args:
            y_preds (numpy.ndarray): The input data.

        Returns:
            The model's final predictions.
        """
        return numpy.exp(y_preds)


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
        link: str = "auto",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            n_bits=n_bits,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

        assert_true(
            link in ["auto", "log", "identity"],
            f"link must be an element of ['auto', 'identity', 'log'], got '{link}'",
        )

        self.power = power
        self.link = link

    def _inverse_link(self, y_preds) -> numpy.ndarray:
        """Apply the link function's inverse on the inputs.

        TweedieRegressor uses either the identity or the exponential function.

        Args:
            y_preds (numpy.ndarray): The input data.

        Returns:
            The model's final predictions.
        """

        if self.link == "auto":

            # Identity link
            if self.power <= 0:
                return y_preds

            # Log link
            return numpy.exp(y_preds)

        if self.link == "log":
            return numpy.exp(y_preds)

        return y_preds
