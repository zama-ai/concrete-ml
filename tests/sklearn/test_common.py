"""Tests common to all sklearn models."""

import inspect
import warnings

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.common.utils import get_model_class
from concrete.ml.pytest.utils import MODELS_AND_DATASETS
from concrete.ml.sklearn import _get_sklearn_all_models


def test_sklearn_args():
    """Check that all arguments from the underlying sklearn model are exposed."""
    test_counter = 0
    for model_class in _get_sklearn_all_models():
        model_class = get_model_class(model_class)

        # For Neural Network models, we manually fix the module parameter to
        # SparseQuantNeuralNetImpl. It is therefore not exposed to the users.
        assert (
            not set(inspect.getfullargspec(model_class.sklearn_model_class).args)
            - set(inspect.getfullargspec(model_class).args)
            - {"module"}
        )
        test_counter += 1

    assert test_counter == 21


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
def test_seed_sklearn(model_class, parameters, load_data):
    """Tests the random_state parameter."""
    x, y = load_data(model_class, **parameters)

    if "random_state" not in inspect.getfullargspec(model_class).args:
        pytest.skip(f"Model class: `{model_class}` has no arguments random_state.")

    random_state_constructor = numpy.random.randint(0, 2**15)
    random_state_user = numpy.random.randint(0, 2**15)

    # First case: user gives his own random_state
    model = model_class(random_state=random_state_constructor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x, y, random_state=random_state_user)

    assert (
        model.random_state == random_state_user and sklearn_model.random_state == random_state_user
    )

    # Second case: user does not give random_state but seeds the constructor
    model = model_class(random_state=random_state_constructor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x, y)

    assert (model.random_state == random_state_constructor) and (
        sklearn_model.random_state == random_state_constructor
    )

    # Third case: user does not provide any seed
    model = model_class(random_state=None)
    assert model.random_state is None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x, y)

    # model.random_state and sklearn_model.random_state should now be seeded with the same value
    assert model.random_state is not None and sklearn_model.random_state is not None
    assert model.random_state == sklearn_model.random_state
