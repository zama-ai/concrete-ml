"""Implement RandomForest models."""
from typing import Any, Dict

import numpy
import sklearn.ensemble

from ..sklearn.tree_to_numpy import tree_to_numpy
from .base import BaseTreeClassifierMixin, BaseTreeEstimatorMixin, BaseTreeRegressorMixin


# pylint: disable=too-many-instance-attributes
class RandomForestClassifier(BaseTreeClassifierMixin):
    """Implements the RandomForest classifier."""

    sklearn_model_class = sklearn.ensemble.RandomForestClassifier
    framework = "sklearn"
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits: int = 6,
        n_estimators=20,
        criterion="gini",
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """Initialize the RandomForestClassifier.

        # noqa: DAR101
        """
        # Call BaseClassifier's __init__ method
        super().__init__(n_bits=n_bits)

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Here, we want to use BaseTreeEstimatorMixin's `post-processing` method as
        # RandomForestClassifier models directly computes probabilities and therefore don't require
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
        metadata["n_estimators"] = self.n_estimators
        metadata["bootstrap"] = self.bootstrap
        metadata["oob_score"] = self.oob_score
        metadata["n_jobs"] = self.n_jobs
        metadata["random_state"] = self.random_state
        metadata["verbose"] = self.verbose
        metadata["warm_start"] = self.warm_start
        metadata["class_weight"] = self.class_weight
        metadata["max_samples"] = self.max_samples
        metadata["criterion"] = self.criterion
        metadata["max_depth"] = self.max_depth
        metadata["min_samples_split"] = self.min_samples_split
        metadata["min_samples_leaf"] = self.min_samples_leaf
        metadata["min_weight_fraction_leaf"] = self.min_weight_fraction_leaf
        metadata["max_features"] = self.max_features
        metadata["max_leaf_nodes"] = self.max_leaf_nodes
        metadata["min_impurity_decrease"] = self.min_impurity_decrease
        metadata["ccp_alpha"] = self.ccp_alpha

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        # Instantiate the model
        obj = RandomForestClassifier(n_bits=metadata["n_bits"])

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
        obj.n_estimators = metadata["n_estimators"]
        obj.bootstrap = metadata["bootstrap"]
        obj.oob_score = metadata["oob_score"]
        obj.n_jobs = metadata["n_jobs"]
        obj.random_state = metadata["random_state"]
        obj.verbose = metadata["verbose"]
        obj.warm_start = metadata["warm_start"]
        obj.class_weight = metadata["class_weight"]
        obj.max_samples = metadata["max_samples"]
        obj.criterion = metadata["criterion"]
        obj.max_depth = metadata["max_depth"]
        obj.min_samples_split = metadata["min_samples_split"]
        obj.min_samples_leaf = metadata["min_samples_leaf"]
        obj.min_weight_fraction_leaf = metadata["min_weight_fraction_leaf"]
        obj.max_features = metadata["max_features"]
        obj.max_leaf_nodes = metadata["max_leaf_nodes"]
        obj.min_impurity_decrease = metadata["min_impurity_decrease"]
        obj.ccp_alpha = metadata["ccp_alpha"]

        return obj


# pylint: disable=too-many-instance-attributes
class RandomForestRegressor(BaseTreeRegressorMixin):
    """Implements the RandomForest regressor."""

    sklearn_model_class = sklearn.ensemble.RandomForestRegressor
    framework = "sklearn"
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits: int = 6,
        n_estimators=20,
        criterion="squared_error",
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """Initialize the RandomForestRegressor.

        # noqa: DAR101
        """
        # Call BaseTreeEstimatorMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

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
        metadata["n_estimators"] = self.n_estimators
        metadata["bootstrap"] = self.bootstrap
        metadata["oob_score"] = self.oob_score
        metadata["n_jobs"] = self.n_jobs
        metadata["random_state"] = self.random_state
        metadata["verbose"] = self.verbose
        metadata["warm_start"] = self.warm_start
        metadata["max_samples"] = self.max_samples
        metadata["criterion"] = self.criterion
        metadata["max_depth"] = self.max_depth
        metadata["min_samples_split"] = self.min_samples_split
        metadata["min_samples_leaf"] = self.min_samples_leaf
        metadata["min_weight_fraction_leaf"] = self.min_weight_fraction_leaf
        metadata["max_features"] = self.max_features
        metadata["max_leaf_nodes"] = self.max_leaf_nodes
        metadata["min_impurity_decrease"] = self.min_impurity_decrease
        metadata["ccp_alpha"] = self.ccp_alpha

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = RandomForestRegressor(n_bits=metadata["n_bits"])

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
        obj.n_estimators = metadata["n_estimators"]
        obj.bootstrap = metadata["bootstrap"]
        obj.oob_score = metadata["oob_score"]
        obj.n_jobs = metadata["n_jobs"]
        obj.random_state = metadata["random_state"]
        obj.verbose = metadata["verbose"]
        obj.warm_start = metadata["warm_start"]
        obj.max_samples = metadata["max_samples"]
        obj.criterion = metadata["criterion"]
        obj.max_depth = metadata["max_depth"]
        obj.min_samples_split = metadata["min_samples_split"]
        obj.min_samples_leaf = metadata["min_samples_leaf"]
        obj.min_weight_fraction_leaf = metadata["min_weight_fraction_leaf"]
        obj.max_features = metadata["max_features"]
        obj.max_leaf_nodes = metadata["max_leaf_nodes"]
        obj.min_impurity_decrease = metadata["min_impurity_decrease"]
        obj.ccp_alpha = metadata["ccp_alpha"]

        return obj
