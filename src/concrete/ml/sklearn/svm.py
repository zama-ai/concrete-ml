"""Implement Support Vector Machine."""
import numpy
import onnx
import sklearn.linear_model

from ..common.check_inputs import check_array_and_assert
from ..onnx.onnx_model_manipulations import clean_graph_after_sigmoid
from .base import SklearnLinearModelMixin


# pylint does not like sklearn uppercase namings
# pylint: disable=invalid-name,too-many-instance-attributes,super-init-not-called
class LinearSVR(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A Regression Support Vector Machine (SVM)."""

    sklearn_alg = sklearn.svm.LinearSVR

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        n_bits=2,
        epsilon=0.0,
        tol=0.0001,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_bits = n_bits
        self._onnx = None
        super().__init__(n_bits=n_bits)

    # pylint: enable=too-many-arguments


class LinearSVC(SklearnLinearModelMixin, sklearn.base.ClassifierMixin):
    """A Classification Support Vector Machine (SVM)."""

    sklearn_alg = sklearn.svm.LinearSVC

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        n_bits=2,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual=True,
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_bits = n_bits
        self._onnx = None
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

    def post_processing(
        self, y_preds: numpy.ndarray, already_dequantized: bool = False
    ) -> numpy.ndarray:
        if not already_dequantized:
            y_preds = super().post_processing(y_preds)
        if y_preds.shape[1] == 1:
            # Sigmoid already applied in the graph
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
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
        y_preds = self.post_processing(y_preds, already_dequantized=True)
        return y_preds

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        X = check_array_and_assert(X)
        y_preds = self.predict_proba(X, execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds
