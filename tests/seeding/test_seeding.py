"""Tests for the torch to numpy module."""

import inspect
import random
import warnings

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import plot_tree

from concrete.ml.pytest.utils import MODELS_AND_DATASETS


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


@pytest.mark.parametrize("random_inputs_1", [numpy.random.randint(0, 2**15, size=20)])
@pytest.mark.parametrize("random_inputs_2", [numpy.random.randint(-1000, 2**15, size=20)])
@pytest.mark.parametrize("random_inputs_3", [numpy.random.randint(-100, 2**15, size=20)])
def test_seed_needing_randomly_seed_arg_1(random_inputs_1, random_inputs_2, random_inputs_3):
    """Test python and numpy seeding for pytest parameters.

    Remark this test needs an extra --randomly-seed argument for reproducibility
    """

    print("Random inputs", random_inputs_1)
    print("Random inputs", random_inputs_2)
    print("Random inputs", random_inputs_3)


@pytest.mark.parametrize("random_inputs_1", [numpy.random.uniform(0, 2**15, size=20)])
@pytest.mark.parametrize("random_inputs_2", [numpy.random.uniform(-1000, 2**15, size=20)])
@pytest.mark.parametrize("random_inputs_3", [numpy.random.uniform(-100, 2**15, size=20)])
def test_seed_needing_randomly_seed_arg_2(random_inputs_1, random_inputs_2, random_inputs_3):
    """Test python and numpy seeding for pytest parameters.

    Remark this test needs an extra --randomly-seed argument for reproducibility
    """

    print("Random inputs", random_inputs_1)
    print("Random inputs", random_inputs_2)
    print("Random inputs", random_inputs_3)


@pytest.mark.parametrize("random_inputs_1", [numpy.random.randint(0, 2**15, size=20)])
@pytest.mark.parametrize("random_inputs_2", [numpy.random.uniform(-1000, 2**15, size=20)])
@pytest.mark.parametrize("random_inputs_3", [numpy.random.randint(-100, 2**15, size=20)])
def test_seed_needing_randomly_seed_arg_3(random_inputs_1, random_inputs_2, random_inputs_3):
    """Test python and numpy seeding for pytest parameters.

    Remark this test needs an extra --randomly-seed argument for reproducibility
    """

    print("Random inputs", random_inputs_1)
    print("Random inputs", random_inputs_2)
    print("Random inputs", random_inputs_3)


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
def test_seed_sklearn(model_class, parameters, load_data, default_configuration):
    """Test seeding of sklearn models"""

    x, y = load_data(model_class, **parameters)

    # Force "random_state": if it was there, it is overwritten; if it was not there, it is added
    model_params = {}
    if "random_state" in inspect.getfullargspec(model_class).args:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    # First case: user gives his own random_state
    model = model_class(**model_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Fit the model
        model, sklearn_model = model.fit_benchmark(x, y)

    lpvoid_ptr_plot_tree = getattr(model, "plot_tree", None)
    if callable(lpvoid_ptr_plot_tree):
        print("model", plot_tree(model.sklearn_model))

    print(f"sklearn_model = {sklearn_model}")

    # Test the determinism of our package (even if the bit-width may be too large)
    try:
        model.compile(x, configuration=default_configuration, show_mlir=True)
    except RuntimeError as err:
        print(err)
    except AssertionError as err:
        print(err)
