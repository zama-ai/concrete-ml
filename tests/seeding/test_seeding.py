"""Tests for the torch to numpy module."""
import random
import warnings
from functools import partial

import numpy
import pytest
from sklearn import tree
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import classifiers, regressors, sanitize_test_and_train_datasets
from concrete.ml.sklearn import (
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    PoissonRegressor,
    Ridge,
    TweedieRegressor,
)


def test_seed_1():
    """Test python and numpy seeding."""

    # Python random
    for _ in range(10):
        print(random.randint(0, 1000))

    # Numpy random
    for _ in range(10):
        print(numpy.random.randint(0, 1000))
        print(numpy.random.uniform(-100, 100, size=(3, 3)))


def test_seed_2():
    """Test python and numpy seeding."""

    # Python random
    for _ in range(20):
        print(random.randint(0, 100))

    # Numpy random
    for _ in range(20):
        print(numpy.random.randint(0, 100))
        print(numpy.random.uniform(-10, 100, size=(3, 3)))


@pytest.mark.parametrize("random_inputs", [numpy.random.randint(0, 2**15, size=20)])
def test_seed_3(random_inputs):
    """Test python and numpy seeding for pytest parameters."""

    print("Random inputs", random_inputs)


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_seed_sklearn(model, parameters, load_data, default_configuration):
    """Test seeding of sklearn regression model using a DecisionTreeRegressor."""

    x, y = load_data(**parameters)
    model_params, x, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)

    # FIXME #2251: some models have not random_state for now. Check if it is expected or not
    if model in [
        Ridge,
        GammaRegressor,
        TweedieRegressor,
        PoissonRegressor,
        ElasticNet,
        Lasso,
        LinearRegression,
    ]:
        return

    # FIXME #2251: some models have not random_state for now. Check if it is expected or not
    if isinstance(model, partial):
        if model.func in [NeuralNetRegressor, NeuralNetClassifier]:
            return

    # Force "random_state": if it was there, it is overwritten; if it was not there, it is added
    model_params["random_state"] = numpy.random.randint(0, 2**15)
    model = model(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x_train, y_train)

    lpvoid_ptr_plot_tree = getattr(model, "plot_tree", None)
    if callable(lpvoid_ptr_plot_tree):
        print("model", tree.plot_tree(model.sklearn_model))

    print("model", sklearn_model)

    # Test the determinism of our package (even if the bitwidth may be too large)
    try:
        model.compile(x, configuration=default_configuration, show_mlir=True)
    except RuntimeError as err:
        print(err)
    except AssertionError as err:
        print(err)
