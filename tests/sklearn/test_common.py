"""Tests common to all sklearn models."""
import warnings
from functools import partial

import numpy
import pytest
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


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_seed_sklearn(model, parameters, load_data):
    """Tests the random_state parameter."""
    x, y = load_data(**parameters)
    _, _, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)
    model_class = model

    random_state_constructor = numpy.random.randint(0, 2**15)
    random_state_user = numpy.random.randint(0, 2**15)

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

    # First case: user gives his own random_state
    model = model_class(random_state=random_state_constructor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x_train, y_train, random_state=random_state_user)

    assert (
        model.random_state == random_state_user and sklearn_model.random_state == random_state_user
    )

    # Second case: user does not give random_state but seeds the constructor
    model = model_class(random_state=random_state_constructor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x_train, y_train)

    assert (model.random_state == random_state_constructor) and (
        sklearn_model.random_state == random_state_constructor
    )

    # Third case: user does not provide any seed
    model = model_class(random_state=None)
    assert model.random_state is None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x_train, y_train)

    # model.random_state and sklearn_model.random_state should now be seeded with the same value
    assert model.random_state is not None and sklearn_model.random_state is not None
    assert model.random_state == sklearn_model.random_state
