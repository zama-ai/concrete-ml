"""Implement DecisionTree models."""

import numpy
import sklearn

from .base import BaseTreeClassifierMixin, BaseTreeEstimatorMixin, BaseTreeRegressorMixin


# pylint: disable-next=too-many-instance-attributes
class DecisionTreeClassifier(BaseTreeClassifierMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    underlying_model_class = sklearn.tree.DecisionTreeClassifier
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

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Here, we want to use BaseTreeEstimatorMixin's `post-processing` method as
        # DecisionTreeClassifier models directly computes probabilities and therefore don't require
        # to apply a sigmoid or softmax in post-processing
        return BaseTreeEstimatorMixin.post_processing(self, y_preds)


# pylint: disable-next=too-many-instance-attributes
class DecisionTreeRegressor(BaseTreeRegressorMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    underlying_model_class = sklearn.tree.DecisionTreeRegressor
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
