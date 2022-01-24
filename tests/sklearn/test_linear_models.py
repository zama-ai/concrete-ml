"""Tests for the sklearn linear models."""
import pytest
from sklearn.datasets import make_regression

from concrete.ml.sklearn import LinearRegression

datasets = [
    pytest.param(
        LinearRegression,
        lambda: make_regression(n_samples=200, n_features=10, random_state=42),
        id="make_regression_10_features",
    ),
    pytest.param(
        LinearRegression,
        lambda: make_regression(n_samples=200, n_features=10, noise=2, random_state=42),
        id="make_regression_features_10_noise_2",
    ),
    pytest.param(
        LinearRegression,
        lambda: make_regression(n_samples=200, n_features=14, n_informative=14, random_state=42),
        id="make_regression_features_14_informative_14",
    ),
    pytest.param(
        LinearRegression,
        lambda: make_regression(
            n_samples=200, n_features=14, n_targets=2, n_informative=14, random_state=42
        ),
        id="make_regression_features_14_informative_14_targets_2",
    ),
]


@pytest.mark.parametrize(
    "alg, load_data",
    datasets,
)
def test_linear_model_compile_run_fhe(load_data, alg, default_compilation_configuration):
    """Tests the sklearn LinearRegression."""

    # Get the dataset
    x, y = load_data()

    # Here we fix n_bits = 2 to make sure the quantized model does not overflow
    # during the compilation.
    model = alg(n_bits=2)
    model, _ = model.fit_benchmark(x, y)

    model.predict(x)

    # Test compilation
    model.compile(x, default_compilation_configuration)

    # Make sure we can predict over a single example in FHE.
    model.predict(x[:1], execute_in_fhe=True)


@pytest.mark.parametrize(
    "alg, load_data",
    datasets,
)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(20, id="20_bits"),
        pytest.param(16, id="16_bits"),
    ],
)
def test_linear_model_quantization(
    alg,
    load_data,
    n_bits,
    check_r2_score,
):
    """Tests the sklearn LinearModel quantization."""

    # Get the dataset
    x, y = load_data()

    model = alg(n_bits=n_bits)
    model, sklearn_model = model.fit_benchmark(x, y)
    y_pred_quantized = model.predict(x)
    y_pred_sklearn = sklearn_model.predict(x)
    check_r2_score(y_pred_sklearn, y_pred_quantized)
