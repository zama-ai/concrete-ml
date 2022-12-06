"""Test tree regressor"""
import warnings
from typing import Any, List

import numpy
import pytest
from sklearn.exceptions import ConvergenceWarning

# FIXME #2320: remaining factorization to be done
from concrete.ml.sklearn import DecisionTreeRegressor, RandomForestRegressor, XGBRegressor


# Get the datasets. The data generation is seeded in load_data.
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


regressor_model_classes = [DecisionTreeRegressor, RandomForestRegressor, XGBRegressor]

multiple_models_datasets: List[Any] = []
models_datasets: List[Any] = []

for regression_model in regressor_model_classes:
    datasets_regression = get_datasets_regression(regression_model)
    multiple_models_datasets += datasets_regression
    models_datasets.append(datasets_regression[0])


@pytest.mark.parametrize("model, parameters", multiple_models_datasets)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
@pytest.mark.parametrize("verbose_compilation", [True, False])
def test_model_compile_run_fhe(
    model,
    parameters,
    use_virtual_lib,
    load_data,
    default_configuration,
    is_vl_only_option,
    verbose_compilation,
):
    """Tests the sklearn regressions."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**parameters)

    # Here we fix n_bits = 2 to make sure the quantized model does not overflow during the
    # compilation.
    model_instantiated = model(n_bits=2)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        if model is XGBRegressor and "n_targets" in parameters and parameters["n_targets"] > 1:
            # XGBRegressor doesn't work with n_targets > 1
            with pytest.raises(AssertionError) as excinfo:
                # Random state should be taken from the method parameter
                model_instantiated.fit_benchmark(x, y)

            assert "n_targets = 1 is the only supported case" in str(excinfo.value)
            return

        model, _ = model_instantiated.fit_benchmark(x, y)

    y_pred = model.predict(x[:1])

    # Test compilation
    model.compile(
        x,
        default_configuration,
        use_virtual_lib=use_virtual_lib,
        verbose_compilation=verbose_compilation,
    )

    # Make sure we can predict over a single example in FHE.
    y_pred_fhe = model.predict(x[:1], execute_in_fhe=True)

    # Check that the output shape is correct
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
def test_model_quantization(model, parameters, n_bits, load_data):
    """Tests quantization of sklearn decision tree regressors."""

    # Get the dataset
    x, y = load_data(**parameters)
    model_instantiated = model(n_bits=n_bits)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        params = {}

        if model is XGBRegressor and "n_targets" in parameters and parameters["n_targets"] > 1:
            # XGBRegressor doesn't work with n_targets > 1
            with pytest.raises(AssertionError) as excinfo:
                # Random state should be taken from the method parameter
                model_instantiated.fit_benchmark(x, y, **params)

            assert "n_targets = 1 is the only supported case" in str(excinfo.value)
            return

        model, sklearn_model = model_instantiated.fit_benchmark(x, y, **params)

    # Check that both models have similar scores
    assert abs(sklearn_model.score(x, y) - model.score(x, y)) < 0.05
