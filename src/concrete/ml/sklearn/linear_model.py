"""Implement sklearn linear model."""
import numpy
import onnx
import sklearn.linear_model

from ..common.check_inputs import check_array_and_assert
from ..onnx.onnx_model_manipulations import clean_graph_after_sigmoid
from .base import SklearnLinearModelMixin


# pylint: disable=invalid-name,too-many-instance-attributes
class LinearRegression(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A linear regression model with FHE."""

    sklearn_alg = sklearn.linear_model.LinearRegression

    def __init__(
        self,
        n_bits=2,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        # FIXME #893
        self.n_bits = n_bits
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)


class LogisticRegression(SklearnLinearModelMixin, sklearn.base.ClassifierMixin):
    """A logistic regression model with FHE."""

    sklearn_alg = sklearn.linear_model.LogisticRegression
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        n_bits=2,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        # FIXME #893
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)

    # pylint: enable=too-many-arguments

    def clean_graph(self, onnx_model: onnx.ModelProto):
        """Clean the graph of the onnx model.

        Args:
            onnx_model (onnx.ModelProto): the onnx model

        Returns:
            onnx.ModelProto: the cleaned onnx model
        """
        onnx_model = clean_graph_after_sigmoid(onnx_model)
        return super().clean_graph(onnx_model)

    def post_processing(
        self, y_preds: numpy.ndarray, already_dequantized: bool = False
    ) -> numpy.ndarray:
        if not already_dequantized:
            y_preds = super().post_processing(y_preds)
        if y_preds.shape[1] == 1:
            # Sigmoid already applied in the graph
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
        else:
            y_preds = numpy.exp(y_preds)
            y_preds = y_preds / numpy.sum(y_preds, axis=1, keepdims=True)
        return y_preds

    def decision_function(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict confidence scores for samples.

        Args:
            X: samples to predict
            execute_in_fhe: if True, the model will be executed in FHE mode

        Returns:
            numpy.ndarray: confidence scores for samples
        """
        y_preds = super().predict(X, execute_in_fhe)
        return y_preds

    def predict_proba(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: samples to predict
            execute_in_fhe: if True, the model will be executed in FHE mode

        Returns:
            numpy.ndarray: class probabilities for samples
        """

        X = check_array_and_assert(X)
        y_preds = self.decision_function(X, execute_in_fhe)
        y_preds = self.post_processing(y_preds, True)
        return y_preds

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        X = check_array_and_assert(X)
        y_preds = self.predict_proba(X, execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds


# pylint: enable=too-many-instance-attributes,invalid-name
