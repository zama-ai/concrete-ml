"""Tests common to all sklearn models."""
import inspect
import warnings
from functools import partial

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.common.utils import get_model_name
from concrete.ml.pytest.utils import classifiers, regressors
from concrete.ml.sklearn.base import (
    get_sklearn_linear_models,
    get_sklearn_neural_net_models,
    get_sklearn_tree_models,
)


def test_sklearn_args():
    """Check that all arguments from the underlying sklearn model are exposed."""
    test_counter = 0
    skipped = []
    for model_class in (
        get_sklearn_linear_models() + get_sklearn_neural_net_models() + get_sklearn_tree_models()
    ):
        if isinstance(model_class, partial):
            model_class = model_class.func
        if hasattr(model_class, "sklearn_alg"):
            assert not set(inspect.getfullargspec(model_class.sklearn_alg).args) - set(
                inspect.getfullargspec(model_class).args
            )
            test_counter += 1
        else:
            skipped.append(model_class.__name__)

    assert test_counter == 16
    assert skipped == ["NeuralNetClassifier", "NeuralNetRegressor"]


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_seed_sklearn(model, parameters, load_data):
    """Tests the random_state parameter."""
    model_class = model
    model_name = get_model_name(model_class)
    x, y = load_data(**parameters, model_name=model_name)

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
