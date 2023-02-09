"""Implement sklearn's Generalized Linear Models (GLM)."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Union

import numpy
import torch
from sklearn.linear_model import GammaRegressor as SklearnGammaRegressor
from sklearn.linear_model import PoissonRegressor as SklearnPoissonRegressor
from sklearn.linear_model import TweedieRegressor as SklearnTweedieRegressor

from concrete.ml import TRUSTED_SKOPS, USE_SKOPS, loads_sklearn
from concrete.ml.quantization.quantizers import UniformQuantizer

from ..common.debugging.custom_assert import assert_true
from ..torch.numpy_module import NumpyModule
from .base import SklearnLinearRegressorMixin
from .torch_modules import _LinearTorchModel


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

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        return self._inverse_link(y_preds)

    def predict(self, X, execute_in_fhe: bool = False) -> numpy.ndarray:
        # Call SklearnLinearModelMixin's predict method
        y_preds = super().predict(X, execute_in_fhe=execute_in_fhe)

        y_preds = self.post_processing(y_preds)
        return y_preds

    # Remove the following method once Hummingbird's latest version is integrated in Concrete-ML
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2080
    def _set_onnx_model(self, test_input: numpy.ndarray):
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        # Initialize the Torch model. Using a small Torch model that reproduces the proper
        # inference is necessary for GLMs. Indeed, the Hummingbird library does not support these
        # models and thus cannot be used to convert them into an ONNX form. Additionally, a
        # NumpyModule accepts a Torch module as an input, making the rest of the structure still
        # usable.
        torch_model = _LinearTorchModel(
            weight=self.sklearn_model.coef_,
            bias=self.sklearn_model.intercept_ if self.fit_intercept else None,
        )

        # Create a NumpyModule from the Torch model
        numpy_module = NumpyModule(
            torch_model,
            dummy_input=torch.from_numpy(test_input[0]),
        )

        # Retrieve the ONNX graph
        self.onnx_model_ = numpy_module.onnx_model

    @abstractmethod
    def _inverse_link(self, y_preds):
        """Apply the link function's inverse on the inputs.

        Args:
            y_preds (numpy.ndarray): The input data.
        """

    def dump_dict(self) -> Dict:
        metadata: Dict[str, Any] = {}
        metadata["cml_dumped_class_name"] = str(type(self).__name__)
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["underlying_model_class"] = self.underlying_model_class
        metadata["post_processing_params"] = self.post_processing_params
        metadata["fhe_circuit"] = self.fhe_circuit
        metadata["input_quantizers"] = [elt.dumps() for elt in self.input_quantizers]
        metadata["_weight_quantizer"] = self._weight_quantizer.dumps()
        metadata["output_quantizers"] = [elt.dumps() for elt in self.output_quantizers]
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_output_scale"] = self._output_scale
        metadata["_output_zero_point"] = self._output_zero_point
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias

        metadata["n_bits"] = self.n_bits
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["solver"] = self.solver
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["warm_start"] = self.warm_start
        metadata["verbose"] = self.verbose
        metadata["onnx_model_"] = self.onnx_model_

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        obj = cls(n_bits=metadata["n_bits"])

        obj.input_quantizers = [UniformQuantizer.loads(elt) for elt in metadata["input_quantizers"]]
        obj._weight_quantizer = UniformQuantizer.loads(metadata["_weight_quantizer"])
        obj.output_quantizers = [
            UniformQuantizer.loads(elt) for elt in metadata["output_quantizers"]
        ]
        loads_sklearn_kwargs = {}
        if USE_SKOPS:
            loads_sklearn_kwargs["trusted"] = TRUSTED_SKOPS
        obj.sklearn_model = loads_sklearn(
            bytes.fromhex(metadata["sklearn_model"]), **loads_sklearn_kwargs
        )

        obj.post_processing_params = metadata["post_processing_params"]
        # Needs to be an array and not a list
        for key, value in obj.post_processing_params.items():
            if isinstance(value, list):
                obj.post_processing_params[key] = numpy.array(value)

        obj.fhe_circuit = metadata["fhe_circuit"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._output_scale = metadata["_output_scale"]
        obj._output_zero_point = metadata["_output_zero_point"]
        if isinstance(obj._output_zero_point, list):
            obj._output_zero_point = numpy.array(obj._output_zero_point)
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]

        obj.n_bits = metadata["n_bits"]
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.solver = metadata["solver"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.warm_start = metadata["warm_start"]
        obj.verbose = metadata["verbose"]
        # pylint: disable-next=protected-access
        obj.onnx_model_ = metadata["onnx_model_"]

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

    underlying_model_class = SklearnPoissonRegressor
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

    def _inverse_link(self, y_preds) -> numpy.ndarray:
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

    underlying_model_class = SklearnGammaRegressor
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

    def _inverse_link(self, y_preds) -> numpy.ndarray:
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

    underlying_model_class = SklearnTweedieRegressor
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

    @property
    def _is_fitted(self):
        return (
            super()._is_fitted
            and self.post_processing_params.get("link", None) is not None
            and self.post_processing_params.get("power", None) is not None
        )

    def _set_post_processing_params(self):
        super()._set_post_processing_params()
        self.post_processing_params.update(
            {
                "link": self.link,
                "power": self.power,
            }
        )

    def _inverse_link(self, y_preds) -> numpy.ndarray:
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
        metadata: Dict[str, Any] = {}
        metadata["cml_dumped_class_name"] = str(type(self).__name__)
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["underlying_model_class"] = self.underlying_model_class
        metadata["post_processing_params"] = self.post_processing_params
        metadata["fhe_circuit"] = self.fhe_circuit
        metadata["input_quantizers"] = [elt.dumps() for elt in self.input_quantizers]
        metadata["_weight_quantizer"] = self._weight_quantizer.dumps()
        metadata["output_quantizers"] = [elt.dumps() for elt in self.output_quantizers]
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_output_scale"] = self._output_scale
        metadata["_output_zero_point"] = self._output_zero_point
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias

        metadata["power"] = self.power
        metadata["link"] = self.link

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        obj = cls(n_bits=metadata["n_bits"])

        obj.input_quantizers = [UniformQuantizer.loads(elt) for elt in metadata["input_quantizers"]]
        obj._weight_quantizer = UniformQuantizer.loads(metadata["_weight_quantizer"])
        obj.output_quantizers = [
            UniformQuantizer.loads(elt) for elt in metadata["output_quantizers"]
        ]
        loads_sklearn_kwargs = {}
        if USE_SKOPS:
            loads_sklearn_kwargs["trusted"] = TRUSTED_SKOPS
        obj.sklearn_model = loads_sklearn(
            bytes.fromhex(metadata["sklearn_model"]), **loads_sklearn_kwargs
        )

        obj.post_processing_params = metadata["post_processing_params"]
        # Needs to be an array and not a list
        for key, value in obj.post_processing_params.items():
            if isinstance(value, list):
                obj.post_processing_params[key] = numpy.array(value)

        obj.fhe_circuit = metadata["fhe_circuit"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._output_scale = metadata["_output_scale"]
        obj._output_zero_point = metadata["_output_zero_point"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]

        obj.power = metadata["power"]
        obj.link = metadata["link"]

        return obj
