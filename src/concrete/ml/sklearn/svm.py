"""Implement Support Vector Machine."""
from typing import Any, Dict

import numpy
import sklearn.linear_model

from .. import TRUSTED_SKOPS, USE_SKOPS, loads_sklearn
from ..quantization.quantizers import UniformQuantizer
from .base import SklearnLinearClassifierMixin, SklearnLinearRegressorMixin


# pylint: disable=invalid-name,too-many-instance-attributes
class LinearSVR(SklearnLinearRegressorMixin):
    """A Regression Support Vector Machine (SVM).

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LinearSVR please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    """

    sklearn_model_class = sklearn.svm.LinearSVR
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        epsilon=0.0,
        tol=0.0001,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        metadata["post_processing_params"] = self.post_processing_params
        metadata["cml_dumped_class_name"] = type(self).__name__

        # Linear
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["sklearn_model_class"] = self.sklearn_model_class
        metadata["fhe_circuit"] = self.fhe_circuit
        metadata["input_quantizers"] = [elt.dumps() for elt in self.input_quantizers]
        metadata["_weight_quantizer"] = self._weight_quantizer.dumps()
        metadata["output_quantizers"] = [elt.dumps() for elt in self.output_quantizers]
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias

        metadata["epsilon"] = self.epsilon
        metadata["tol"] = self.tol
        metadata["C"] = self.C
        metadata["loss"] = self.loss
        metadata["fit_intercept"] = self.fit_intercept
        metadata["intercept_scaling"] = self.intercept_scaling
        metadata["dual"] = self.dual
        metadata["verbose"] = self.verbose
        metadata["random_state"] = self.random_state
        metadata["max_iter"] = self.max_iter

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        obj = LinearSVR()
        obj.post_processing_params = metadata["post_processing_params"]

        # Load the underlying fitted model
        loads_sklearn_kwargs = {}
        if USE_SKOPS:
            loads_sklearn_kwargs["trusted"] = TRUSTED_SKOPS
        obj.sklearn_model = loads_sklearn(
            bytes.fromhex(metadata["sklearn_model"]), **loads_sklearn_kwargs
        )

        # Linear
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model_class = metadata["sklearn_model_class"]
        obj.fhe_circuit = metadata["fhe_circuit"]
        obj.input_quantizers = [UniformQuantizer.loads(elt) for elt in metadata["input_quantizers"]]
        obj.output_quantizers = [
            UniformQuantizer.loads(elt) for elt in metadata["output_quantizers"]
        ]
        obj._weight_quantizer = UniformQuantizer.loads(metadata["_weight_quantizer"])
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]

        obj.epsilon = metadata["epsilon"]
        obj.tol = metadata["tol"]
        obj.C = metadata["C"]
        obj.loss = metadata["loss"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.intercept_scaling = metadata["intercept_scaling"]
        obj.dual = metadata["dual"]
        obj.verbose = metadata["verbose"]
        obj.random_state = metadata["random_state"]
        obj.max_iter = metadata["max_iter"]

        return obj


class LinearSVC(SklearnLinearClassifierMixin):
    """A Classification Support Vector Machine (SVM).

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LinearSVC please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    """

    sklearn_model_class = sklearn.svm.LinearSVC
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual=True,
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        # Call BaseClassifier's __init__ method
        super().__init__(n_bits=n_bits)

        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        metadata["post_processing_params"] = self.post_processing_params
        metadata["cml_dumped_class_name"] = type(self).__name__

        # Classifier
        metadata["classes_"] = self.target_classes_
        metadata["n_classes_"] = self.n_classes_
        metadata["cml_dumped_class_name"] = type(self).__name__

        # Linear
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["sklearn_model_class"] = self.sklearn_model_class
        metadata["fhe_circuit"] = self.fhe_circuit
        metadata["input_quantizers"] = [elt.dumps() for elt in self.input_quantizers]
        metadata["_weight_quantizer"] = self._weight_quantizer.dumps()
        metadata["output_quantizers"] = [elt.dumps() for elt in self.output_quantizers]
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias

        metadata["penalty"] = self.penalty
        metadata["loss"] = self.loss
        metadata["dual"] = self.dual
        metadata["tol"] = self.tol
        metadata["C"] = self.C
        metadata["multi_class"] = self.multi_class
        metadata["fit_intercept"] = self.fit_intercept
        metadata["intercept_scaling"] = self.intercept_scaling
        metadata["class_weight"] = self.class_weight
        metadata["verbose"] = self.verbose
        metadata["random_state"] = self.random_state
        metadata["max_iter"] = self.max_iter

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        obj = LinearSVC()
        obj.post_processing_params = metadata["post_processing_params"]

        # Load the underlying fitted model
        loads_sklearn_kwargs = {}
        if USE_SKOPS:
            loads_sklearn_kwargs["trusted"] = TRUSTED_SKOPS
        obj.sklearn_model = loads_sklearn(
            bytes.fromhex(metadata["sklearn_model"]), **loads_sklearn_kwargs
        )

        # Classifier
        obj.target_classes_ = numpy.array(metadata["classes_"])
        obj.n_classes_ = metadata["n_classes_"]

        # Linear
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model_class = metadata["sklearn_model_class"]
        obj.fhe_circuit = metadata["fhe_circuit"]
        obj.input_quantizers = [UniformQuantizer.loads(elt) for elt in metadata["input_quantizers"]]
        obj.output_quantizers = [
            UniformQuantizer.loads(elt) for elt in metadata["output_quantizers"]
        ]
        obj._weight_quantizer = UniformQuantizer.loads(metadata["_weight_quantizer"])
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]

        obj.penalty = metadata["penalty"]
        obj.loss = metadata["loss"]
        obj.dual = metadata["dual"]
        obj.tol = metadata["tol"]
        obj.C = metadata["C"]
        obj.multi_class = metadata["multi_class"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.intercept_scaling = metadata["intercept_scaling"]
        obj.class_weight = metadata["class_weight"]
        obj.verbose = metadata["verbose"]
        obj.random_state = metadata["random_state"]
        obj.max_iter = metadata["max_iter"]

        return obj


# pylint: enable=invalid-name,too-many-instance-attributes
