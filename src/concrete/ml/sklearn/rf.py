"""Implements RandomForest models."""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy
import sklearn.ensemble

from ..common.debugging.custom_assert import assert_true
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
        n_bits: int = 7,
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
            self, qX, framework="sklearn", output_n_bits=self.n_bits
        )
        return self

    def quantize_input(self, X: numpy.ndarray):
        """Quantize the input.

        Args:
            X (numpy.ndarray): the input

        Returns:
            the quantized input
        """
        qX = numpy.zeros_like(X)
        # Quantize using the learned quantization parameters for each feature
        for i, q_x_ in enumerate(self.q_x_byfeatures):
            qX[:, i] = q_x_.update_values(X[:, i])
        return qX.astype(numpy.int32)

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
        y_preds = self.predict_proba(X, execute_in_fhe=execute_in_fhe, *args, **kwargs)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds

    def predict_proba(
        self, X: numpy.ndarray, *args, execute_in_fhe: bool = False, **kwargs
    ) -> numpy.ndarray:
        """Predict the probabilities.

        Note: RandomForestClassifier already outputs probabilities thus we don't need to
            apply softmax or sigmoid.

        Args:
            X (numpy.ndarray): The input data.
            args: args for super().predict
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.
            kwargs: kwargs for super().predict

        Returns:
            numpy.ndarray: The predicted probabilities.
        """
        assert_true(len(args) == 0, f"Unsupported **args parameters {args}")
        assert_true(len(kwargs) == 0, f"Unsupported **kwargs parameters {kwargs}")
        assert_true(execute_in_fhe is False, "execute_in_fhe is not supported")
        # mypy
        assert self._tensor_tree_predict is not None
        qX = self.quantize_input(X)
        y_preds = self._tensor_tree_predict(qX)[0]
        y_preds = self.q_y.update_quantized_values(y_preds)
        assert_true(y_preds.ndim > 1, "y_preds should be a 2D array")
        y_preds = numpy.transpose(y_preds)
        y_preds = numpy.sum(y_preds, axis=-1)
        # Make sure that numpy.sum(y_preds, axis=1) = 1
        assert_true(
            bool(numpy.alltrue(numpy.abs(numpy.sum(y_preds, axis=1)) - 1 < 1e-6)),
            "y_preds should sum to 1",
        )
        # Apply softmax not needed as RF already has sum(trees_outputs) = 1
        return y_preds

    def fit_benchmark(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        *args,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[RandomForestClassifier, sklearn.ensemble._forest.RandomForestClassifier]:
        """Fit the sklearn RandomForestClassifier and the FHE RandomForestClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            random_state (Optional[Union[int, numpy.random.RandomState, None]]):
                The random state. Defaults to None.
            *args: args for super().fit
            **kwargs: kwargs for super().fit

        Returns:
            Tuple[RandomForestClassifier, sklearn.ensemble._forest.RandomForestClassifier]:
                The FHE and RandomForestClassifier.
        """
        # Make sure the random_state is set or both algorithms will diverge
        # due to randomness in the training.
        if random_state is not None:
            self.init_args["random_state"] = random_state
        elif self.random_state is not None:
            self.init_args["random_state"] = self.random_state
        else:
            self.init_args["random_state"] = numpy.random.randint(0, 2**15)

        # Train the sklearn model without X quantized
        sklearn_model = sklearn.ensemble._forest.RandomForestClassifier(**self.init_args)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        super().__init__(**self.init_args)
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model
