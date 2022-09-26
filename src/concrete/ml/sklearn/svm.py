"""Implement Support Vector Machine."""
import sklearn.linear_model

from .base import SklearnLinearClassifierMixin, SklearnLinearModelMixin


# pylint: disable=invalid-name,too-many-instance-attributes
class LinearSVR(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A Regression Support Vector Machine (SVM)."""

    sklearn_alg = sklearn.svm.LinearSVR

    # pylint: disable-next=too-many-arguments
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
        # FIXME: Figure out how to add scikit-learn documentation into our object #893
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
    """A Classification Support Vector Machine (SVM)."""

    sklearn_alg = sklearn.svm.LinearSVC

    # pylint: disable-next=too-many-arguments
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


# pylint: enable=invalid-name,too-many-instance-attributes
