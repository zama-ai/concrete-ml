"""Implement Support Vector Machine."""
from __future__ import annotations

import sklearn.linear_model

from .linear_model import SklearnLinearModelMixin


class LinearSVR(SklearnLinearModelMixin, sklearn.svm.LinearSVR):
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
        super().__init__(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.n_bits = n_bits

    # pylint: enable=too-many-arguments
