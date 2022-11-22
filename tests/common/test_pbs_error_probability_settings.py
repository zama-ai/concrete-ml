"""Tests for the sklearn linear models."""
import warnings
from functools import partial

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import classifiers, regressors
from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    RandomForestClassifier,
    XGBClassifier,
    XGBRegressor,
)


# FIXME #2203: to be reused, renamed and moved
def clean_this(model, x, y):
    """This function needs to be moved in shared.py and used everywhere."""

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    if isinstance(model, partial):
        if model.func in [NeuralNetClassifier, NeuralNetRegressor]:
            model_params = model.keywords
            # Change module__input_dim to be the same as the input dimension
            model_params["module__input_dim"] = x_train.shape[1]
            # qnns require float32 as input
            x = x.astype(numpy.float32)
            x_train = x_train.astype(numpy.float32)
            x_test = x_test.astype(numpy.float32)

            if model.func is NeuralNetRegressor:
                # Reshape y_train and y_test if 1d (regression for neural nets)
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1).astype(numpy.float32)
    elif model in [XGBClassifier, RandomForestClassifier, XGBRegressor]:
        model_params = {
            "n_estimators": 5,
            "max_depth": 2,
            "random_state": numpy.random.randint(0, 2**15),
        }
    elif model is DecisionTreeClassifier:
        model_params = {"max_depth": 2, "random_state": numpy.random.randint(0, 2**15)}
    elif model in [LogisticRegression]:
        model_params = {"random_state": numpy.random.randint(0, 2**15)}
    else:
        model_params = {}

    return model_params, x, y, x_train, y_train


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
@pytest.mark.parametrize("kwargs", [{"p_error": 0.03}, {"p_error": 0.04, "global_p_error": None}])
def test_p_error(model, parameters, kwargs, load_data):
    """Testing with p_error."""
    x, y = load_data(**parameters)

    model_params, x, _, x_train, y_train = clean_this(model, x, y)

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

    model_params, x, _, x_train, y_train = clean_this(model, x, y)

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

    model_params, x, _, x_train, y_train = clean_this(model, x, y)

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

    model_params, x, _, x_train, y_train = clean_this(model, x, y)

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
