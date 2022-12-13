"""Tests common to all sklearn models."""
import inspect
import warnings
from functools import partial

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import (
    classifier_models,
    classifiers,
    regressor_models,
    regressors,
    sanitize_test_and_train_datasets,
)


def test_sklearn_args():
    """Check that all arguments from the underlying sklearn model are exposed."""
    test_counter = 0
    skipped = []
    for model_class in classifier_models + regressor_models:
        if isinstance(model_class, partial):
            model_class = model_class.func
        if hasattr(model_class, "sklearn_alg"):
            assert not set(inspect.getfullargspec(model_class.sklearn_alg).args) - set(
                inspect.getfullargspec(model_class).args
            )
            test_counter += 1
        else:
            skipped.append(model_class.__name__)

    assert test_counter == 20
    assert skipped == ["NeuralNetClassifier", "NeuralNetRegressor"]


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_seed_sklearn(model, parameters, load_data):
    """Tests the random_state parameter."""
    x, y = load_data(**parameters)
    _, _, _, x_train, y_train, _ = sanitize_test_and_train_datasets(model, x, y)
    model_class = model

    if "random_state" not in inspect.getfullargspec(model_class).args:
        pytest.skip(f"Model class: `{model_class}` has no arguments random_state.")

    random_state_constructor = numpy.random.randint(0, 2**15)
    random_state_user = numpy.random.randint(0, 2**15)

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
