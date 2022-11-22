"""Tests for the sklearn linear models."""
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import classifiers, regressors, sanitize_test_and_train_datasets


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
@pytest.mark.parametrize("kwargs", [{"p_error": 0.03}, {"p_error": 0.04, "global_p_error": None}])
def test_p_error(model, parameters, kwargs, load_data):
    """Testing with p_error."""
    x, y = load_data(**parameters)
    model_params, x, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)

    clf = model(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        clf.fit(x_train, y_train)

    clf.compile(x_train, verbose_compilation=True, **kwargs)

    # FIXME: waiting for https://github.com/zama-ai/concrete-numpy-internal/issues/1737 and
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1738
    #
    # Will check that we have
    #   3.000000e-02 error per pbs call
    # in Optimizer config


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
@pytest.mark.parametrize(
    "kwargs", [{"global_p_error": 0.025}, {"global_p_error": 0.036, "p_error": None}]
)
def test_global_p_error(model, parameters, kwargs, load_data):
    """Testing with global_p_error."""

    x, y = load_data(**parameters)
    x_train = x[:-1]
    y_train = y[:-1]

    model_params, x, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)

    clf = model(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        clf.fit(x_train, y_train)

    clf.compile(x_train, verbose_compilation=True, **kwargs)

    # FIXME: waiting for https://github.com/zama-ai/concrete-numpy-internal/issues/1737 and
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1738
    #
    # Will check that we have
    #   2.500000e-02 error per circuit call
    # in Optimizer config


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_global_p_error_and_p_error_together(model, parameters, load_data):
    """Testing with both p_error and global_p_error."""

    x, y = load_data(**parameters)
    x_train = x[:-1]
    y_train = y[:-1]

    model_params, x, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)

    clf = model(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        clf.fit(x_train, y_train)

    with pytest.raises(ValueError) as excinfo:
        clf.compile(x_train, verbose_compilation=True, global_p_error=0.025, p_error=0.017)

    assert "Please only set one of (p_error, global_p_error) values" in str(excinfo.value)


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_default(model, parameters, load_data):
    """Testing with default."""

    x, y = load_data(**parameters)
    x_train = x[:-1]
    y_train = y[:-1]

    model_params, x, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)

    clf = model(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        clf.fit(x_train, y_train)

    clf.compile(x_train, verbose_compilation=True)

    # FIXME: waiting for https://github.com/zama-ai/concrete-numpy-internal/issues/1737 and
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1738
    #
    # Will check that we have CN default
    # in Optimizer config
