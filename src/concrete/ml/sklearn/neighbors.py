"""Implement sklearn linear model."""
from typing import Any, Dict

import numpy
import sklearn.linear_model

from ..common.debugging.custom_assert import assert_true
from .base import SklearnKNeighborsClassifierMixin


# pylint: disable=invalid-name,too-many-instance-attributes
class KNeighborsClassifier(SklearnKNeighborsClassifierMixin):
    """A k-nearest classifier model with FHE.

    Parameters:
        n_bits (int): Number of bits to quantize the model. The value will be used for quantizing
            inputs and X_fit. Default to 3.

    For more details on KNeighborsClassifier please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    sklearn_model_class = sklearn.neighbors.KNeighborsClassifier
    _is_a_public_cml_model = True

    def __init__(
        self,
        n_bits=2,
        n_neighbors=3,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        # Call SklearnKNeighborsClassifierMixin's __init__ method
        super().__init__(n_bits=n_bits)

        assert_true(
            algorithm in ["brute", "auto"], f"Algorithm = `{algorithm}` is not supported in FHE."
        )
        assert_true(
            not callable(metric), "The KNeighborsClassifier does not support custom metrics."
        )
        assert_true(
            p == 2 and metric == "minkowski",
            "Only `L2` norm is supported with `p=2` and `metric = 'minkowski'`",
        )

        self._y: numpy.ndarray
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.weights = weights

    def dump_dict(self) -> Dict[str, Any]:
        assert self._q_fit_X_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_q_fit_X_quantizer"] = self._q_fit_X_quantizer
        metadata["_q_fit_X"] = self._q_fit_X
        metadata["_y"] = self._y

        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["post_processing_params"] = self.post_processing_params
        metadata["cml_dumped_class_name"] = type(self).__name__

        # scikit-learn
        metadata["sklearn_model_class"] = self.sklearn_model_class
        metadata["n_neighbors"] = self.n_neighbors
        metadata["algorithm"] = self.algorithm
        metadata["weights"] = self.weights
        metadata["leaf_size"] = self.leaf_size
        metadata["p"] = self.p
        metadata["metric"] = self.metric
        metadata["metric_params"] = self.metric_params
        metadata["n_jobs"] = self.n_jobs

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = KNeighborsClassifier()

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._q_fit_X_quantizer = metadata["_q_fit_X_quantizer"]
        obj._q_fit_X = metadata["_q_fit_X"]
        obj._y = metadata["_y"]

        obj.onnx_model_ = metadata["onnx_model_"]

        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.n_neighbors = metadata["n_neighbors"]
        obj.weights = metadata["weights"]
        obj.algorithm = metadata["algorithm"]
        obj.p = metadata["p"]
        obj.metric = metadata["metric"]
        obj.metric_params = metadata["metric_params"]
        obj.n_jobs = metadata["n_jobs"]
        return obj
