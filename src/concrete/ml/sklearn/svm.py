"""Implement Support Vector Machine."""
import sklearn.linear_model

from .base import SklearnLinearClassifierMixin, SklearnLinearModelMixin


# pylint: disable=invalid-name,too-many-instance-attributes
class LinearSVR(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A Regression Support Vector Machine (SVM).

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LinearSVR please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    """

    sklearn_alg = sklearn.svm.LinearSVR
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
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


class LinearSVC(SklearnLinearClassifierMixin, sklearn.base.ClassifierMixin):
    """A Classification Support Vector Machine (SVM).

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LinearSVC please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    """

    sklearn_alg = sklearn.svm.LinearSVC
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
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


# pylint: enable=invalid-name,too-many-instance-attributes
