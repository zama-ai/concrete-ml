"""Implements RandomForest models."""
from typing import Any, Callable, List, Optional

import numpy
import sklearn.ensemble
from custom_inherit import doc_inherit

from ..quantization import QuantizedArray
from .base import BaseTreeEstimatorMixin
from .tree_to_numpy import tree_to_numpy


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-ancestors
# Hummingbird needs to see the protected _forest class
# pylint: disable=protected-access,too-many-instance-attributes
class RandomForestClassifier(
    BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator
):
    """Implements the RandomForest classifier."""

    sklearn_alg = sklearn.ensemble.RandomForestClassifier
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    q_y: QuantizedArray
    _tensor_tree_predict: Optional[Callable]
    sklearn_model: Any

    # pylint: disable=too-many-arguments

    @doc_inherit(
        sklearn.ensemble._forest.RandomForestClassifier.__init__, style="google_with_merge"
    )
    def __init__(
        self,
        n_bits: int = 6,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
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
        # FIXME #893
        BaseTreeEstimatorMixin.__init__(self, n_bits=n_bits)
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

    # pylint: enable=too-many-arguments

    #  pylint: disable=arguments-differ
    def fit(self, X: numpy.ndarray, y: numpy.ndarray, **kwargs) -> "RandomForestClassifier":
        """Fit the RandomForestClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            **kwargs: args for super().fit

        Returns:
            RandomForestClassifier: The RandomForestClassifier.
        """
        # mypy
        assert self.n_bits is not None

        qX = numpy.zeros_like(X)
        self.q_x_byfeatures = []

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i])
            self.q_x_byfeatures.append(q_x_)
            qX[:, i] = q_x_.qvalues.astype(numpy.int32)

        # Initialize the sklearn model
        params = self.get_params()
        params.pop("n_bits", None)

        self.sklearn_model = self.sklearn_alg(**params)

        self.sklearn_model.fit(qX, y, **kwargs)

        # Tree ensemble inference to numpy
        self._tensor_tree_predict, self.q_y = tree_to_numpy(
            self.sklearn_model,
            qX,
            framework="sklearn",
            output_n_bits=self.n_bits,
            use_workaround_for_transpose=True,
        )
        return self

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict the target values.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return BaseTreeEstimatorMixin.predict(self, X, execute_in_fhe=execute_in_fhe)

    def predict_proba(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict the probabilities.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.

        Returns:
            numpy.ndarray: The predicted probabilities.
        """
        return BaseTreeEstimatorMixin.predict_proba(self, X, execute_in_fhe=execute_in_fhe)

    # pylint: enable=protected-access,too-many-instance-attributes
