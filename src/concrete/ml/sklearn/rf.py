"""Implements RandomForest models."""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy
import sklearn.ensemble

from ..quantization import QuantizedArray
from .base import BaseTreeEstimatorMixin
from .tree_to_numpy import tree_to_numpy


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-ancestors
# Hummingbird needs to see the protected _forest class
# pylint: disable=protected-access
class RandomForestClassifier(
    sklearn.ensemble._forest.RandomForestClassifier, BaseTreeEstimatorMixin
):
    """Implements the RandomForest classifier."""

    sklearn_alg = sklearn.ensemble.RandomForestClassifier
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    q_y: QuantizedArray
    _tensor_tree_predict: Optional[Callable]

    def __init__(
        self,
        n_bits: int = 6,
        max_depth: Optional[int] = 15,
        n_estimators: Optional[int] = 100,
        **kwargs: Any,
    ):
        """Initialize the RandomForestClassifier.

        Args:
            n_bits (int): The number of bits to use. Defaults to 7.
            max_depth (Optional[int]): The maximum depth of the tree. Defaults to 15.
            n_estimators (Optional[int]): The number of estimators. Defaults to 100.
            **kwargs: args for super().__init__
        """
        sklearn.ensemble.RandomForestClassifier.__init__(
            self,
            max_depth=max_depth,
            n_estimators=n_estimators,
            **kwargs,
        )
        BaseTreeEstimatorMixin.__init__(self, n_bits=n_bits)
        self.init_args = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            **kwargs,
        }

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

        super().fit(qX, y, **kwargs)

        # Tree ensemble inference to numpy
        self._tensor_tree_predict, self.q_y = tree_to_numpy(
            self,
            qX,
            framework="sklearn",
            output_n_bits=self.n_bits,
            use_workaround_for_transpose=True,
        )
        return self

    def predict(
        self, X: numpy.ndarray, *args, execute_in_fhe: bool = False, **kwargs
    ) -> numpy.ndarray:
        """Predict the target values.

        Args:
            X (numpy.ndarray): The input data.
            args: args for super().predict
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.
            kwargs: kwargs for super().predict

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return BaseTreeEstimatorMixin.predict(
            self, X, *args, execute_in_fhe=execute_in_fhe, **kwargs
        )

    def predict_proba(
        self, X: numpy.ndarray, *args, execute_in_fhe: bool = False, **kwargs
    ) -> numpy.ndarray:
        """Predict the probabilities.

        Args:
            X (numpy.ndarray): The input data.
            args: args for super().predict
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.
            kwargs: kwargs for super().predict

        Returns:
            numpy.ndarray: The predicted probabilities.
        """
        return BaseTreeEstimatorMixin.predict_proba(
            self, X, *args, execute_in_fhe=execute_in_fhe, **kwargs
        )
