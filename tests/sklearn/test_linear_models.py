"""Tests for the sklearn linear models."""
import warnings
from functools import partial
from typing import Any, List

import numpy
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# FIXME #2320: remaining factorization to be done
from concrete.ml.sklearn import (
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
    TweedieRegressor,
)

regressor_model_classes = [
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LinearSVR,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
]

classifier_model_classes = [LogisticRegression, LinearSVC]


# Get the datasets. The data generation is seeded in load_data.
def get_datasets_regression(model_class):
    """Return tests to apply to a regression model."""

    regression_datasets = [
        pytest.param(
            model_class,
            {
                "dataset": "regression",
                "strictly_positive": model_class
                in [GammaRegressor, PoissonRegressor, TweedieRegressor],
                "n_features": 10,
            },
            id=f"make_regression_features_10_{model_class.__name__}",
        ),
        pytest.param(
            model_class,
            {
                "dataset": "regression",
                "strictly_positive": model_class
                in [GammaRegressor, PoissonRegressor, TweedieRegressor],
                "n_features": 10,
                "noise": 2,
            },
            id=f"make_regression_features_10_noise_2_{model_class.__name__}",
        ),
        pytest.param(
            model_class,
            {
                "dataset": "regression",
                "strictly_positive": model_class
                in [GammaRegressor, PoissonRegressor, TweedieRegressor],
                "n_features": 14,
                "n_informative": 14,
            },
            id=f"make_regression_features_14_informative_14_{model_class.__name__}",
        ),
    ]

    # LinearSVR, PoissonRegressor, GammaRegressor and TweedieRegressor do not support multi targets
    if model_class not in [LinearSVR, PoissonRegressor, GammaRegressor, TweedieRegressor]:
        regression_datasets += [
            pytest.param(
                model_class,
                {
                    "dataset": "regression",
                    "n_features": 14,
                    "n_informative": 14,
                    "n_targets": 2,
                },
                id=f"make_regression_features_14_informative_14_targets_2_{model_class.__name__}",
            )
        ]

    # TweedieRegressor has some additional specific tests
    if model_class == TweedieRegressor:
        tweedie_data_parameters = {
            "dataset": "regression",
            "strictly_positive": True,
            "n_features": 10,
        }

        links_powers = [("auto", 0.0), ("auto", 2.8), ("log", 1.0), ("identity", 0.0)]

        for link, power in links_powers:
            regression_datasets += [
                pytest.param(
                    partial(model_class, link=link, power=power),
                    tweedie_data_parameters,
                    id=(
                        "make_regression_features_10"
                        f"_{model_class.__name__}_link_{link}_power_{power}"
                    ),
                )
            ]

    # LinearSVR has some additional specific tests
    if model_class == LinearSVR:
        model_partial = partial(model_class, dual=False, loss="squared_epsilon_insensitive")

        regression_datasets += [
            pytest.param(
                partial(
                    model_partial,
                    fit_intercept=False,
                ),
                {
                    "dataset": "regression",
                    "n_features": 10,
                },
                id=f"make_regression_features_10_fit_intercept_false_{model_class.__name__}",
            ),
            pytest.param(
                partial(
                    model_partial,
                    fit_intercept=True,
                ),
                {
                    "dataset": "regression",
                    "n_features": 10,
                },
                id=f"make_regression_features_10_fit_intercept_false_{model_class.__name__}",
            ),
            pytest.param(
                partial(
                    model_partial,
                    fit_intercept=True,
                    intercept_scaling=1000,
                ),
                {
                    "dataset": "regression",
                    "n_features": 10,
                },
                id=(
                    "make_regression_features_10_fit_intercept_false_intercept_scaling_1000_"
                    f"{model_class.__name__}"
                ),
            ),
        ]

    return regression_datasets


def get_datasets_classification(model_class):
    """Return tests to apply to a classification model."""
    classifier_datasets = [
        pytest.param(
            model_class,
            {
                "dataset": "classification",
                "n_samples": 200,
                "class_sep": 2,
                "n_features": 10,
            },
            id=f"make_classification_features_10_{model_class.__name__}",
        ),
        pytest.param(
            model_class,
            {
                "dataset": "classification",
                "n_samples": 200,
                "class_sep": 2,
                "n_features": 14,
            },
            id=f"make_classification_features_14_{model_class.__name__}",
        ),
        pytest.param(
            model_class,
            {
                "dataset": "classification",
                "n_samples": 200,
                "n_features": 14,
                "n_clusters_per_class": 1,
                "class_sep": 2,
                "n_classes": 4,
            },
            id=f"make_classification_features_14_classes_4_{model_class.__name__}",
        ),
    ]

    return classifier_datasets


multiple_models_datasets: List[Any] = []
models_datasets: List[Any] = []

for regression_model in regressor_model_classes:
    datasets_regression = get_datasets_regression(regression_model)
    multiple_models_datasets += datasets_regression
    models_datasets.append(datasets_regression[0])

for classifier_model in classifier_model_classes:
    datasets_classification = get_datasets_classification(classifier_model)
    multiple_models_datasets += datasets_classification
    models_datasets.append(datasets_classification[0])


@pytest.mark.parametrize("model_class, data_parameters", multiple_models_datasets)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("use_virtual_lib", [True, False])
@pytest.mark.parametrize("verbose_compilation", [True, False])
# pylint: disable-next=too-many-arguments
def test_linear_model_compile_run_fhe(
    model_class,
    data_parameters,
    fit_intercept,
    use_virtual_lib,
    load_data,
    default_configuration,
    is_vl_only_option,
    check_r2_score,
    check_accuracy,
    verbose_compilation,
    check_circuit_has_no_tlu,
):
    """Tests the sklearn regressions."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**data_parameters)

    # Define the model's hyperparameters
    model_hyperparameters = {"n_bits": 8, "fit_intercept": fit_intercept}

    # Instantiate the model
    model = model_class(**model_hyperparameters)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, sklearn_model = model.fit_benchmark(x, y)

    y_pred = model.predict(x)

    # Test compilation
    fhe_circuit = model.compile(
        x,
        default_configuration,
        use_virtual_lib=use_virtual_lib,
        verbose_compilation=verbose_compilation,
    )

    # Check that no TLUs are found within the MLIR
    check_circuit_has_no_tlu(fhe_circuit)

    y_pred_sklearn = sklearn_model.predict(x)
    y_pred_fhe = model.predict(x, execute_in_fhe=True)

    # If the model is a classifier, check that accuracies are similar
    if model_class in classifier_model_classes:
        check_accuracy(y_pred_sklearn, y_pred_fhe)

    # If the model is a regressor, check that R2 scores are similar
    else:
        check_r2_score(y_pred_sklearn, y_pred_fhe)

    # Check that the output shapes are correct
    assert y_pred.shape == y_pred_fhe.shape, "Quantized clear and FHE outputs have different shapes"


@pytest.mark.parametrize("model_class, data_parameters", multiple_models_datasets)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(20, id="20_bits"),
        pytest.param(16, id="16_bits"),
    ],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_model_quantization(
    model_class,
    data_parameters,
    n_bits,
    fit_intercept,
    load_data,
    check_r2_score,
    check_accuracy,
):
    """Tests the sklearn LinearModel quantization."""

    # Get the dataset
    x, y = load_data(**data_parameters)

    # Define the model's hyperparameters
    model_hyperparameters = {"n_bits": n_bits, "fit_intercept": fit_intercept}

    model = model_class(**model_hyperparameters)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        params = {}
        if model_class in classifier_model_classes:
            params["random_state"] = numpy.random.randint(0, 2**15)
            model.set_params(**params)

            # Random state should be taken from the class variable
            model, sklearn_model = model.fit_benchmark(x, y)

        # Random state should be taken from the method parameter
        model, sklearn_model = model.fit_benchmark(x, y, **params)

    # If the model is a classifier
    if model_class in classifier_model_classes:

        # Check that accuracies are similar
        y_pred_quantized = model.predict(x)
        y_pred_sklearn = sklearn_model.predict(x)
        check_accuracy(y_pred_sklearn, y_pred_quantized)

        # If the model is a LinearSVC model, compute its predicted confidence score
        # This is done separately as scikit-learn doesn't provide a predict_proba method for
        # LinearSVC models
        if isinstance(model, LinearSVC):
            y_pred_quantized = model.decision_function(x)
            y_pred_sklearn = sklearn_model.decision_function(x)

        # Else, compute the model's predicted probabilities
        else:
            y_pred_quantized = model.predict_proba(x)
            y_pred_sklearn = sklearn_model.predict_proba(x)

    # If the model is a regressor, compute its predictions
    else:
        y_pred_quantized = model.predict(x)
        y_pred_sklearn = sklearn_model.predict(x)

    # Check that predictions, probabilities or confidence scores are similar using the R2 score
    check_r2_score(y_pred_sklearn, y_pred_quantized)


@pytest.mark.parametrize("model_class, data_parameters", models_datasets)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_pipeline_sklearn(model_class, data_parameters, fit_intercept, load_data):
    """Tests that the linear models work well within sklearn pipelines."""
    x, y = load_data(**data_parameters)

    # Define the model's hyperparameters
    model_hyperparameters = {"n_bits": 2, "fit_intercept": fit_intercept}

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2, random_state=numpy.random.randint(0, 2**15))),
            ("scaler", StandardScaler()),
            ("model", model_class(**model_hyperparameters)),
        ]
    )

    # Do a grid search to find the best hyperparameters
    param_grid = {
        "model__n_bits": [2, 3],
    }
    grid_search = GridSearchCV(pipe_cv, param_grid, error_score="raise", cv=3)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        grid_search.fit(x, y)
