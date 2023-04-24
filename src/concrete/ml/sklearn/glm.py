"""Implement sklearn's Generalized Linear Models (GLM)."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Union

import numpy
import sklearn.linear_model

from ..common.debugging.custom_assert import assert_true
from ..common.utils import FheMode
from ..onnx.onnx_model_manipulations import clean_graph_after_node_op_type
from .base import Data, SklearnLinearRegressorMixin


# pylint: disable-next=too-many-instance-attributes
class _GeneralizedLinearRegressor(SklearnLinearRegressorMixin):
    """Regression via a penalized Generalized Linear Model (GLM) with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs: number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.
    """

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 8,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def _clean_graph(self) -> None:
        assert self.onnx_model_ is not None, self._is_not_fitted_error_message()

        # Remove any operators following gemm, as they will be done in the clear in post-processing
        # In particular, this includes the exponential operator
        clean_graph_after_node_op_type(self.onnx_model_, node_op_type="Gemm")
        super()._clean_graph()

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        return self._inverse_link(y_preds)

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Call SklearnLinearModelMixin's predict method
        y_preds = super().predict(X, fhe=fhe)

        y_preds = self.post_processing(y_preds)
        return y_preds

    @abstractmethod
    def _inverse_link(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply the link function's inverse on the inputs.

        Args:
            y_preds (numpy.ndarray): The input data.
        """

    def dump_dict(self) -> Dict:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # Scikit-Learn
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["solver"] = self.solver
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["warm_start"] = self.warm_start
        metadata["verbose"] = self.verbose

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.solver = metadata["solver"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.warm_start = metadata["warm_start"]
        obj.verbose = metadata["verbose"]

        return obj


class PoissonRegressor(_GeneralizedLinearRegressor):
    """A Poisson regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on PoissonRegressor please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
    """

    sklearn_model_class = sklearn.linear_model.PoissonRegressor
    _is_a_public_cml_model = True

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 8,
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

    def _inverse_link(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        return numpy.exp(y_preds)


class GammaRegressor(_GeneralizedLinearRegressor):
    """A Gamma regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on GammaRegressor please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html
    """

    sklearn_model_class = sklearn.linear_model.GammaRegressor
    _is_a_public_cml_model = True

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 8,
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

    def _inverse_link(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        return numpy.exp(y_preds)


# pylint: disable-next=too-many-instance-attributes
class TweedieRegressor(_GeneralizedLinearRegressor):
    """A Tweedie regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on TweedieRegressor please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
    """

    sklearn_model_class = sklearn.linear_model.TweedieRegressor
    _is_a_public_cml_model = True

    def __init__(
        self,
        *,
        n_bits: Union[int, dict] = 8,
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

    def _set_post_processing_params(self):
        super()._set_post_processing_params()
        self.post_processing_params.update(
            {
                "link": self.link,
                "power": self.power,
            }
        )

    def _inverse_link(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        if self.post_processing_params["link"] == "auto":

            # Identity link
            if self.post_processing_params["power"] <= 0:
                return y_preds

            # Log link
            return numpy.exp(y_preds)

        if self.post_processing_params["link"] == "log":
            return numpy.exp(y_preds)

        return y_preds

    def dump_dict(self) -> Dict:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params
        metadata["power"] = self.power
        metadata["link"] = self.link

        # Scikit-Learn
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["solver"] = self.solver
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["warm_start"] = self.warm_start
        metadata["verbose"] = self.verbose

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete-ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.power = metadata["power"]
        obj.link = metadata["link"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.solver = metadata["solver"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.warm_start = metadata["warm_start"]
        obj.verbose = metadata["verbose"]

        return obj
