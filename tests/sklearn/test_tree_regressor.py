"""Test tree regressor"""
import warnings

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.sklearn import DecisionTreeRegressor


def get_datasets_regression(model):
    """Return tests to apply to a regression model."""

    regression_datasets = [
        pytest.param(
            model,
            {
                "dataset": "regression",
                "strictly_positive": False,
                "n_features": 10,
            },
            id=f"make_regression_features_10_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "regression",
                "strictly_positive": False,
                "n_features": 10,
                "noise": 2,
            },
            id=f"make_regression_features_10_noise_2_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "regression",
                "strictly_positive": False,
                "n_features": 14,
                "n_informative": 14,
            },
            id=f"make_regression_features_14_informative_14_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "regression",
                "n_features": 14,
                "n_informative": 14,
                "n_targets": 2,
            },
            id=f"make_regression_features_14_informative_14_targets_2_{model.__name__}",
        ),
    ]

    return regression_datasets


multiple_models_datasets = get_datasets_regression(DecisionTreeRegressor)
models_datasets = [multiple_models_datasets[0]]


@pytest.mark.parametrize("model, parameters", multiple_models_datasets)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_model_compile_run_fhe(
    model, parameters, use_virtual_lib, load_data, default_configuration, is_vl_only_option
):
    """Tests the sklearn regressions."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**parameters)

    # Here we fix n_bits = 2 to make sure the quantized model does not overflow during the
    # compilation.
    model = model(n_bits=2)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, _ = model.fit_benchmark(x, y)

    y_pred = model.predict(x[:1])

    # Test compilation
    model.compile(x, default_configuration, use_virtual_lib=use_virtual_lib)

    # Make sure we can predict over a single example in FHE.
    y_pred_fhe = model.predict(x[:1], execute_in_fhe=True)

    # Check that the ouput shape is correct
    assert y_pred_fhe.shape == y_pred.shape
    assert numpy.array_equal(y_pred_fhe, y_pred)


@pytest.mark.parametrize("model, parameters", multiple_models_datasets)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(20, id="20_bits"),
        pytest.param(16, id="16_bits"),
    ],
)
def test_model_quantization(
    model,
    parameters,
    n_bits,
    load_data,
    check_r2_score,
):
    """Tests quantization of sklearn decision tree regressors."""

    # Get the dataset
    x, y = load_data(**parameters)
    model = model(n_bits=n_bits)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        params = {}
        # Random state should be taken from the method parameter
        model, sklearn_model = model.fit_benchmark(x, y, **params)

    y_pred_sklearn = model.predict(x)
    y_pred_quantized = sklearn_model.predict(x)

    check_r2_score(y_pred_sklearn, y_pred_quantized)
