"""Implement the sklearn tree models."""
from typing import Callable, Optional

import sklearn
from concrete.numpy.compilation.circuit import Circuit

from ..quantization.quantizers import QuantizedArray
from .base import BaseTreeClassifierMixin, BaseTreeRegressorMixin


# pylint: disable-next=too-many-instance-attributes
class DecisionTreeClassifier(BaseTreeClassifierMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    sklearn_alg = sklearn.tree.DecisionTreeClassifier
    q_x_byfeatures: list
    fhe_tree: Circuit
    _tensor_tree_predict: Optional[Callable]
    q_y: QuantizedArray
    class_mapping_: Optional[dict]
    n_classes_: int
    framework: str = "sklearn"

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

        BaseTreeClassifierMixin.__init__(self, n_bits=n_bits)
        self.q_x_byfeatures = []
        self.n_bits = n_bits
        self.fhe_tree = None
        self.class_mapping_ = None


# pylint: disable-next=too-many-instance-attributes
class DecisionTreeRegressor(BaseTreeRegressorMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    sklearn_alg = sklearn.tree.DecisionTreeRegressor
    q_x_byfeatures: list
    fhe_tree: Circuit
    _tensor_tree_predict: Optional[Callable]
    q_y: QuantizedArray
    framework: str = "sklearn"

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

        BaseTreeRegressorMixin.__init__(self, n_bits=n_bits)
        self.q_x_byfeatures = []
        self.n_bits = n_bits
        self.fhe_tree = None
        self.class_mapping_ = None
