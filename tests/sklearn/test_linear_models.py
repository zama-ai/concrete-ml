"""Tests for the sklearn linear models."""
import warnings
from functools import partial
from typing import Any, List

import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from concrete.ml.sklearn import (
    GammaRegressor,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    PoissonRegressor,
    TweedieRegressor,
)

regression_models = [
    LinearRegression,
    LinearSVR,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
]
classifier_models = [LogisticRegression, LinearSVC]


def get_datasets_regression(model):
    """Return tests to apply to a regression model."""

    regression_datasets = [
        pytest.param(
            model,
            {
                "dataset": "regression",
                "strictly_positive": model in [GammaRegressor, PoissonRegressor, TweedieRegressor],
                "n_features": 10,
            },
            id=f"make_regression_features_10_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "regression",
                "strictly_positive": model in [GammaRegressor, PoissonRegressor, TweedieRegressor],
                "n_features": 10,
                "noise": 2,
            },
            id=f"make_regression_features_10_noise_2_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "regression",
                "strictly_positive": model in [GammaRegressor, PoissonRegressor, TweedieRegressor],
                "n_features": 14,
                "n_informative": 14,
            },
            id=f"make_regression_features_14_informative_14_{model.__name__}",
        ),
    ]

    # LinearSVR, PoissonRegressor, GammaRegressor and TweedieRegressor do not support multi targets
    if model not in [LinearSVR, PoissonRegressor, GammaRegressor, TweedieRegressor]:
        regression_datasets += [
            pytest.param(
                model,
                {
                    "dataset": "regression",
                    "n_features": 14,
                    "n_informative": 14,
                    "n_targets": 2,
                },
                id=f"make_regression_features_14_informative_14_targets_2_{model.__name__}",
            )
        ]

    # TweedieRegressor has some additional specific tests
    if model == TweedieRegressor:
        tweedie_parameters = {
            "dataset": "regression",
            "strictly_positive": True,
            "n_features": 10,
        }
        links = ["auto", "auto", "log", "identity"]
        powers = [0.0, 2.8, 1.0, 0.0]

        for link, power in zip(links, powers):
            regression_datasets += [
                pytest.param(
                    partial(model, link=link, power=power),
                    tweedie_parameters,
                    id=f"make_regression_features_10_{model.__name__}_link_{link}_power_{power}",
                )
            ]

    # if model == LinearSVR:
    #     model = partial(model, dual=False, loss="squared_epsilon_insensitive")

    #     regression_datasets += [
    #         # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/420
    #         pytest.param(
    #             partial(
    #                 model,
    #                 fit_intercept=False,
    #             ),
    #             {
    #                 "n_samples": 200,
    #                 "n_features": 10,
    #                 "random_state": numpy.random.randint(0, 2**15),
    #             },
    #             id=f"make_regression_fit_intercept_false_{model.__name__}",
    #         ),
    #         # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/421
    #         pytest.param(
    #             partial(
    #                 model,
    #                 fit_intercept=True,
    #             ),
    #             {
    #                 "n_samples": 200,
    #                 "n_features": 10,
    #                 "random_state": numpy.random.randint(0, 2**15)
    #             },
    #             id=f"make_regression_fit_intercept_true_{model.__name__}",
    #         ),
    #         # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/421
    #         pytest.param(
    #             partial(
    #                 model,
    #                 fit_intercept=True,
    #                 intercept_scaling=1000,
    #             ),
    #             {
    #                 "n_samples": 200,
    #                 "n_features": 10,
    #                 "random_state": numpy.random.randint(0, 2**15),
    #             },
    #             id=f"make_regression_fit_intercept_true_intercept_scaling_1000_{model.__name__}",
    #         ),
    #     ]

    return regression_datasets


def get_datasets_classification(model):
    """Return tests to apply to a classification model."""
    classifier_datasets = [
        pytest.param(
            model,
            {
                "dataset": "classification",
                "n_samples": 200,
                "class_sep": 2,
                "n_features": 10,
                "random_state": 42,
            },
            id=f"make_classification_features_10_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "classification",
                "n_samples": 200,
                "class_sep": 2,
                "n_features": 14,
                "random_state": 42,
            },
            id=f"make_classification_features_14_informative_14_{model.__name__}",
        ),
        pytest.param(
            model,
            {
                "dataset": "classification",
                "n_samples": 200,
                "n_features": 14,
                "n_clusters_per_class": 1,
                "class_sep": 2,
                "n_classes": 4,
                "random_state": 42,
            },
            id=f"make_classification_features_14_informative_14_classes_4_{model.__name__}",
        ),
    ]

    return classifier_datasets


multiple_models_datasets: List[Any] = []
models_datasets: List[Any] = []

for regression_model in regression_models:
    datasets_regression = get_datasets_regression(regression_model)
    multiple_models_datasets += datasets_regression
    models_datasets.append(datasets_regression[0])

for classifier_model in classifier_models:
    datasets_classification = get_datasets_classification(classifier_model)
    multiple_models_datasets += datasets_classification
    models_datasets.append(datasets_classification[0])


@pytest.mark.parametrize("model, parameters", multiple_models_datasets)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_linear_model_compile_run_fhe(
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


@pytest.mark.parametrize("model, parameters", multiple_models_datasets)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(20, id="20_bits"),
        pytest.param(16, id="16_bits"),
    ],
)
def test_linear_model_quantization(
    model,
    parameters,
    n_bits,
    load_data,
    check_r2_score,
    check_accuracy,
):
    """Tests the sklearn LinearModel quantization."""

    # Get the dataset
    x, y = load_data(**parameters)

    model = model(n_bits=n_bits)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, sklearn_model = model.fit_benchmark(x, y)

    if model._estimator_type == "classifier":  # pylint: disable=protected-access
        # Classification models

        # Check that accuracies are similar
        y_pred_quantized = model.predict(x)
        y_pred_sklearn = sklearn_model.predict(x)
        check_accuracy(y_pred_sklearn, y_pred_quantized)

        if isinstance(model, LinearSVC):  # pylint: disable=no-else-return
            # Test disabled as it our version of decision_function is not
            # the same as sklearn (TODO issue #494)
            # LinearSVC does not implement predict_proba
            # y_pred_quantized = model.decision_function(x)
            # y_pred_sklearn = sklearn_model.decision_function(x)
            return
        else:
            # Check that probabilities are similar
            y_pred_quantized = model.predict_proba(x)
            y_pred_sklearn = sklearn_model.predict_proba(x)
    else:
        # Regression models

        # Check that class prediction are similar
        y_pred_quantized = model.predict(x)
        y_pred_sklearn = sklearn_model.predict(x)

    check_r2_score(y_pred_sklearn, y_pred_quantized)


@pytest.mark.parametrize("model, parameters", models_datasets)
def test_pipeline_sklearn(model, parameters, load_data):
    """Tests that the linear models work well within sklearn pipelines."""
    x, y = load_data(**parameters)

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", StandardScaler()),
            ("model", model()),
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
