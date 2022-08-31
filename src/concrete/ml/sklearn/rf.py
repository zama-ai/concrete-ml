"""Implements RandomForest models."""
from typing import Any, Callable, List, Optional

import sklearn.ensemble

from ..quantization import QuantizedArray
from .base import BaseTreeClassifierMixin, BaseTreeRegressorMixin


# pylint: disable=too-many-instance-attributes
class RandomForestClassifier(BaseTreeClassifierMixin):
    """Implements the RandomForest classifier."""

    sklearn_alg = sklearn.ensemble.RandomForestClassifier
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    q_y: QuantizedArray
    _tensor_tree_predict: Optional[Callable]
    sklearn_model: Any
    framework: str = "sklearn"

    # pylint: disable=too-many-arguments,protected-access

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

        # FIXME #893
        BaseTreeClassifierMixin.__init__(self, n_bits=n_bits)
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


# pylint: disable=too-many-instance-attributes
class RandomForestRegressor(BaseTreeRegressorMixin):
    """Implements the RandomForest regressor."""

    sklearn_alg = sklearn.ensemble.RandomForestRegressor
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    q_y: QuantizedArray
    _tensor_tree_predict: Optional[Callable]
    sklearn_model: Any
    framework: str = "sklearn"

    # pylint: disable=too-many-arguments,protected-access

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

        # FIXME #893
        BaseTreeRegressorMixin.__init__(self, n_bits=n_bits)
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
