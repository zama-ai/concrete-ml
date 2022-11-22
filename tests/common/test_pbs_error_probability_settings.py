"""Tests for the sklearn linear models."""
import warnings

from inspect import signature
from concrete.ml.torch.compile import compile_torch_model

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import classifiers, regressors, sanitize_test_and_train_datasets

from torch import nn

INPUT_OUTPUT_FEATURE = [5, 10]


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self, input_output, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=input_output)
        self.act_f = activation_function()
        self.fc2 = nn.Linear(in_features=input_output, out_features=input_output)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.act_f(out)
        out = self.fc2(out)

        return out


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


@pytest.mark.parametrize(
    "model",
    [pytest.param(FC)],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_error": 0.03},
        {"p_error": 0.04, "global_p_error": None},
        {"global_p_error": 0.025},
        {"global_p_error": 0.036, "p_error": None},
        {},
        {"global_p_error": 0.038, "p_error": 0.39},
    ],
)
def test_config_torch(model, kwargs, load_data):
    """Testing with p_error and global_p_error configs with torch models."""

    input_output_feature = 5
    n_examples = 50

    torch_model = model(input_output_feature, activation_function=nn.ReLU)

    num_inputs = len(signature(torch_model.forward).parameters)

    # Create random input
    inputset = (
        tuple(
            numpy.random.uniform(-100, 100, size=(n_examples, input_output_feature))
            for _ in range(num_inputs)
        )
        if num_inputs > 1
        else numpy.random.uniform(-100, 100, size=(n_examples, input_output_feature))
    )

    if (
        "global_p_error" in kwargs
        and "p_error" in kwargs
        and kwargs["global_p_error"] is not None
        and kwargs["p_error"] is not None
    ):
        with pytest.raises(ValueError) as excinfo:
            compile_torch_model(
                torch_model,
                inputset,
                n_bits={"model_inputs": 2, "model_outputs": 2, "op_inputs": 2, "op_weights": 2},
                use_virtual_lib=False,
                verbose_compilation=True,
                **kwargs
            )

        assert "Please only set one of (p_error, global_p_error) values" in str(excinfo.value)
    else:
        compile_torch_model(
            torch_model,
            inputset,
            n_bits={"model_inputs": 2, "model_outputs": 2, "op_inputs": 2, "op_weights": 2},
            use_virtual_lib=False,
            verbose_compilation=True,
            **kwargs
        )

    # FIXME: waiting for https://github.com/zama-ai/concrete-numpy-internal/issues/1737 and
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1738
    #
    # Will check that we have expected probabilities
