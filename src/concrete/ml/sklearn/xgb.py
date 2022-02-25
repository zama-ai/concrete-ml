"""Implements XGBoost models."""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy
import xgboost.sklearn

from ..common.debugging.custom_assert import assert_true
from ..quantization import QuantizedArray
from .tree_to_numpy import tree_to_numpy


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name
class XGBClassifier(xgboost.sklearn.XGBClassifier):
    """Implements the XGBoost classifier."""

    q_x_byfeatures: List[QuantizedArray]
    n_bits: Optional[int]
    q_y: QuantizedArray
    _tensor_tree_predict: Callable

    def __init__(
        self,
        n_bits: Optional[int] = 7,
        max_depth: Optional[int] = 3,
        learning_rate: Optional[float] = 0.1,
        n_estimators: Optional[int] = 20,
        objective: Optional[str] = "binary:logistic",
        use_label_encoder: Optional[bool] = False,
        base_score: Optional[float] = 0.5,
        verbosity=0,
        **kwargs: Any,
    ):
        """Initialize the XGBoostClassifier.

        Args:
            n_bits (Optional[int]): The number of bits to use. Defaults to 7.
            max_depth (Optional[int]): The maximum depth of the tree. Defaults to 3.
            learning_rate (Optional[float]): The learning rate. Defaults to 0.1.
            n_estimators (Optional[int]): The number of estimators. Defaults to 20.
            objective (Optional[str]): The objective function to use. Defaults to "binary:logistic".
            use_label_encoder (Optional[bool]): Whether to use the label encoder. Defaults to False.
            base_score (Optional[float]): The base score. Defaults to 0.5.
            verbosity (int): Verbosity level. Defaults to 0.
            **kwargs: args for super().__init__
        """
        assert_true(
            base_score == 0.5, "base_score != 0.5 is not yet supported. Please use base_score=0.5"
        )
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            use_label_encoder=use_label_encoder,
            verbosity=verbosity,
            **kwargs,
        )
        self.init_args = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "objective": objective,
            "use_label_encoder": use_label_encoder,
            "verbosity": verbosity,
            **kwargs,
        }
        self.n_bits = n_bits

    #  pylint: disable=arguments-differ
    def fit(self, X: numpy.ndarray, y: numpy.ndarray, **kwargs) -> "XGBClassifier":
        """Fit the XGBoostClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            **kwargs: args for super().fit

        Returns:
            XGBoostClassifier: The XGBoostClassifier.
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
        # Have to ignore mypy (Can't assign to a method)
        self._tensor_tree_predict, self.q_y = tree_to_numpy(  # type: ignore
            self, qX, framework="xgboost", output_n_bits=self.n_bits
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

        qX = self.quantize_input(X)
        y_preds = self._tensor_tree_predict(qX)[0]
        y_preds = self.q_y.update_quantized_values(y_preds)
        y_preds = numpy.squeeze(y_preds)
        assert_true(y_preds.ndim > 1, "y_preds should be a 2D array")
        y_preds = numpy.transpose(y_preds)
        y_preds = numpy.sum(y_preds, axis=1, keepdims=True)
        y_preds = 1.0 / (1.0 + numpy.exp(-y_preds))
        y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
        return y_preds

    def fit_benchmark(
        self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs
    ) -> Tuple[XGBClassifier, xgboost.sklearn.XGBClassifier]:
        """Fit the sklearn XGBoostClassifier and the FHE XGBoostClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: args for super().fit
            **kwargs: kwargs for super().fit

        Returns:
            Tuple[XGBoostClassifier, xgboost.sklearn.XGBoostClassifier]:
                The FHE and XGBoostClassifier.
        """
        # Train the sklearn model without X quantized

        sklearn_model = xgboost.sklearn.XGBClassifier(**self.init_args)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model
