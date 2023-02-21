"""Tests with Torch tensors as inputs."""
import warnings

import numpy
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import sklearn_models_and_datasets
from concrete.ml.sklearn import NeuralNetRegressor


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
def test_torch_tensors(model_class, parameters, load_data):
    """Tests that we can use Torch tensors as inputs to fit and predict."""
    model = model_class()

    # Some models use a bit of randomness while fitting under scikit-learn, making the
    # outputs always different after each fit. In order to avoid that problem, their random_state
    # parameter needs to be fixed each time the test is ran.
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    x, y = load_data(model_class, **parameters)

    # Turn data and targets to Torch tensors
    x, y = torch.tensor(x), torch.tensor(y)

    # Skorch requires its regressor's targets to be 2D, even if there's only one target dimension
    if isinstance(model, NeuralNetRegressor):
        y = y.reshape((-1, 1))

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)
        model.predict(x)
