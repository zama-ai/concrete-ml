"""Implement DecisionTree models."""
from typing import Any, Dict

import numpy
import sklearn.tree

from ..sklearn.tree_to_numpy import tree_to_numpy
from .base import BaseTreeClassifierMixin, BaseTreeEstimatorMixin, BaseTreeRegressorMixin


# pylint: disable-next=too-many-instance-attributes
class DecisionTreeClassifier(BaseTreeClassifierMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    sklearn_model_class = sklearn.tree.DecisionTreeClassifier
    framework = "sklearn"
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha: float = 0.0,
        n_bits: int = 6,
    ):
        """Initialize the DecisionTreeClassifier.

        # noqa: DAR101

        """
        # Call BaseClassifier's __init__ method
        super().__init__(n_bits=n_bits)

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def __getattr__(self, attr: str):
        # We directly expose the following methods as they are commonly used with decision trees
        # and only represent their topological structure
        if (
            attr in ["get_depth", "get_n_leaves"]
            and getattr(self, "sklearn_model", None) is not None
        ):
            return getattr(self.sklearn_model, attr)

        return super().__getattr__(attr)

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Here, we want to use BaseTreeEstimatorMixin's `post-processing` method as
        # DecisionTreeClassifier models directly computes probabilities and therefore don't require
        # to apply a sigmoid or softmax in post-processing
        return BaseTreeEstimatorMixin.post_processing(self, y_preds)

    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["framework"] = self.framework
        metadata["post_processing_params"] = self.post_processing_params

        # Classifier
        metadata["target_classes_"] = self.target_classes_
        metadata["n_classes_"] = self.n_classes_

        # Scikit-Learn
        metadata["criterion"] = self.criterion
        metadata["splitter"] = self.splitter
        metadata["max_depth"] = self.max_depth
        metadata["min_samples_split"] = self.min_samples_split
        metadata["min_samples_leaf"] = self.min_samples_leaf
        metadata["min_weight_fraction_leaf"] = self.min_weight_fraction_leaf
        metadata["max_features"] = self.max_features
        metadata["max_leaf_nodes"] = self.max_leaf_nodes
        metadata["class_weight"] = self.class_weight
        metadata["random_state"] = self.random_state
        metadata["min_impurity_decrease"] = self.min_impurity_decrease
        metadata["ccp_alpha"] = self.ccp_alpha

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete-ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.framework = metadata["framework"]
        obj._tree_inference, obj.output_quantizers, obj.onnx_model_ = tree_to_numpy(
            obj.sklearn_model,
            numpy.zeros((len(obj.input_quantizers),))[None, ...],
            framework=obj.framework,
            output_n_bits=obj.n_bits,
        )
        obj.post_processing_params = metadata["post_processing_params"]

        # Classifier
        obj.target_classes_ = metadata["target_classes_"]
        obj.n_classes_ = metadata["n_classes_"]

        # Scikit-Learn
        obj.criterion = metadata["criterion"]
        obj.splitter = metadata["splitter"]
        obj.max_depth = metadata["max_depth"]
        obj.min_samples_split = metadata["min_samples_split"]
        obj.min_samples_leaf = metadata["min_samples_leaf"]
        obj.min_weight_fraction_leaf = metadata["min_weight_fraction_leaf"]
        obj.max_features = metadata["max_features"]
        obj.max_leaf_nodes = metadata["max_leaf_nodes"]
        obj.class_weight = metadata["class_weight"]
        obj.random_state = metadata["random_state"]
        obj.min_impurity_decrease = metadata["min_impurity_decrease"]
        obj.ccp_alpha = metadata["ccp_alpha"]

        return obj


# pylint: disable-next=too-many-instance-attributes
class DecisionTreeRegressor(BaseTreeRegressorMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    sklearn_model_class = sklearn.tree.DecisionTreeRegressor
    framework = "sklearn"
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        n_bits: int = 6,
    ):
        """Initialize the DecisionTreeRegressor.

        # noqa: DAR101

        """
        # Call BaseTreeEstimatorMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def __getattr__(self, attr: str):
        # We directly expose the following methods as they are commonly used with decision trees
        # and only represent their topological structure
        if (
            attr in ["get_depth", "get_n_leaves"]
            and getattr(self, "sklearn_model", None) is not None
        ):
            return getattr(self.sklearn_model, attr)

        return super().__getattr__(attr)

    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["framework"] = self.framework
        metadata["post_processing_params"] = self.post_processing_params

        # Scikit-Learn
        metadata["criterion"] = self.criterion
        metadata["splitter"] = self.splitter
        metadata["max_depth"] = self.max_depth
        metadata["min_samples_split"] = self.min_samples_split
        metadata["min_samples_leaf"] = self.min_samples_leaf
        metadata["min_weight_fraction_leaf"] = self.min_weight_fraction_leaf
        metadata["max_features"] = self.max_features
        metadata["max_leaf_nodes"] = self.max_leaf_nodes
        metadata["random_state"] = self.random_state
        metadata["min_impurity_decrease"] = self.min_impurity_decrease
        metadata["ccp_alpha"] = self.ccp_alpha

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete-ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.framework = metadata["framework"]
        obj._tree_inference, obj.output_quantizers, obj.onnx_model_ = tree_to_numpy(
            obj.sklearn_model,
            numpy.zeros((len(obj.input_quantizers),))[None, ...],
            framework=obj.framework,
            output_n_bits=obj.n_bits,
        )
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.criterion = metadata["criterion"]
        obj.splitter = metadata["splitter"]
        obj.max_depth = metadata["max_depth"]
        obj.min_samples_split = metadata["min_samples_split"]
        obj.min_samples_leaf = metadata["min_samples_leaf"]
        obj.min_weight_fraction_leaf = metadata["min_weight_fraction_leaf"]
        obj.max_features = metadata["max_features"]
        obj.max_leaf_nodes = metadata["max_leaf_nodes"]
        obj.random_state = metadata["random_state"]
        obj.min_impurity_decrease = metadata["min_impurity_decrease"]
        obj.ccp_alpha = metadata["ccp_alpha"]

        return obj
