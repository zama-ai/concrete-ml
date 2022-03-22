"""Tests for the sklearn linear models."""
import warnings
from typing import Any, List

import numpy
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from concrete.ml.sklearn import LinearRegression, LinearSVC, LinearSVR, LogisticRegression


def get_datasets_regression(reg_model_orig):
    """Return tests to apply to a regression model."""
    reg_model_string = reg_model_orig.__name__

    reg_model = reg_model_orig

    ans = [
        pytest.param(
            reg_model,
            lambda: make_regression(
                n_samples=200, n_features=10, random_state=numpy.random.randint(0, 2**15)
            ),
            id=f"make_regression_10_features_{reg_model_string}",
        ),
        pytest.param(
            reg_model,
            lambda: make_regression(
                n_samples=200, n_features=10, noise=2, random_state=numpy.random.randint(0, 2**15)
            ),
            id=f"make_regression_features_10_noise_2_{reg_model_string}",
        ),
        pytest.param(
            reg_model,
            lambda: make_regression(
                n_samples=200,
                n_features=14,
                n_informative=14,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id=f"make_regression_features_14_informative_14_{reg_model_string}",
        ),
    ]

    # LinearSVR does not support multi targets
    if reg_model_orig != LinearSVR:
        ans += [
            pytest.param(
                reg_model,
                lambda: make_regression(
                    n_samples=200,
                    n_features=14,
                    n_targets=2,
                    n_informative=14,
                    random_state=numpy.random.randint(0, 2**15),
                ),
                id=f"make_regression_features_14_informative_14_targets_2_{reg_model_string}",
            ),
        ]

    # if reg_model_orig == LinearSVR:
    #     reg_model = partial(reg_model, dual=False, loss="squared_epsilon_insensitive")
    #
    #     ans += [
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/420
    # pytest.param(
    #     partial(
    #         reg_model,
    #         fit_intercept=False,
    #     ),
    #     lambda: make_regression(n_samples=200, n_features=10,
    #       random_state=numpy.random.randint(0, 2**15)),
    #     id=f"make_regression_fit_intercept_false_{reg_model_string}",
    # ),
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/421
    # pytest.param(
    #     partial(
    #         reg_model,
    #         fit_intercept=True,
    #     ),
    #     lambda: make_regression(n_samples=200, n_features=10,
    #       random_state=numpy.random.randint(0, 2**15)),
    #     id=f"make_regression_fit_intercept_true_{reg_model_string}",
    # ),
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/421
    # pytest.param(
    #     partial(
    #         reg_model,
    #         fit_intercept=True,
    #         intercept_scaling=1000,
    #     ),
    #     lambda: make_regression(
    #         n_samples=200,
    #         n_features=10,
    #         random_state=numpy.random.randint(0, 2**15),
    #     ),
    #     id=f"make_regression_fit_intercept_true_intercept_scaling_1000_{reg_model_string}",
    # ),
    # ]

    return ans


def get_datasets_classification(class_model_orig):
    """Return tests to apply to a classification model."""
    class_model_string = class_model_orig.__name__

    class_model = class_model_orig

    ans = [
        pytest.param(
            class_model,
            lambda: make_classification(n_samples=200, class_sep=2, n_features=10, random_state=42),
            id=f"make_classification_10_features_{class_model_string}",
        ),
        pytest.param(
            class_model,
            lambda: make_classification(n_samples=200, class_sep=2, n_features=14, random_state=42),
            id=f"make_classification_features_14_informative_14_{class_model_string}",
        ),
        pytest.param(
            class_model,
            lambda: make_classification(
                n_samples=200,
                n_features=14,
                n_clusters_per_class=1,
                class_sep=2,
                n_classes=4,
                random_state=42,
            ),
            id=f"make_classification_features_14_informative_14_classes_4_{class_model_string}",
        ),
    ]

    return ans


datasets_regression: List[Any] = []

for one_regression_model in [LinearRegression, LinearSVR]:
    datasets_regression += get_datasets_regression(one_regression_model)

datasets_classification: List[Any] = []

for one_classifier_model in [LogisticRegression, LinearSVC]:
    datasets_regression += get_datasets_classification(one_classifier_model)


@pytest.mark.parametrize(
    "alg, load_data",
    datasets_regression + datasets_classification,
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_linear_model_compile_run_fhe(
    load_data, alg, use_virtual_lib, default_compilation_configuration
):
    """Tests the sklearn regressions."""

    # Get the dataset
    x, y = load_data()

    # Here we fix n_bits = 2 to make sure the quantized model does not overflow
    # during the compilation.
    model = alg(n_bits=2)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, _ = model.fit_benchmark(x, y)

    y_pred = model.predict(x[:1])

    # Test compilation
    model.compile(x, default_compilation_configuration, use_virtual_lib=use_virtual_lib)

    # Make sure we can predict over a single example in FHE.
    y_pred_fhe = model.predict(x[:1], execute_in_fhe=True)

    # Check that the ouput shape is correct
    assert y_pred_fhe.shape == y_pred.shape


@pytest.mark.parametrize(
    "alg, load_data",
    datasets_regression + datasets_classification,
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
    check_accuracy,
):
    """Tests the sklearn LinearModel quantization."""

    # Get the dataset
    x, y = load_data()

    model = alg(n_bits=n_bits)

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


def test_double_fit():
    """Tests that calling fit multiple times gives the same results"""
    x, y = make_classification()
    model = DecisionTreeClassifier()

    # First fit
    model.fit(x, y)
    y_pred_one = model.predict(x)

    # Second fit
    model.fit(x, y)
    y_pred_two = model.predict(x)

    assert numpy.array_equal(y_pred_one, y_pred_two)


@pytest.mark.parametrize(
    "alg",
    [
        pytest.param(LinearRegression),
        pytest.param(LogisticRegression),
        pytest.param(LinearSVR),
        pytest.param(LinearSVC),
    ],
)
def test_pipeline_sklearn(alg):
    """Tests that the linear models work well within sklearn pipelines."""
    x, y = make_classification(n_features=10, n_informative=2, n_redundant=0, n_classes=2)
    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", StandardScaler()),
            ("alg", alg()),
        ]
    )
    # Do a grid search to find the best hyperparameters
    param_grid = {
        "alg__n_bits": [2, 3],
    }
    grid_search = GridSearchCV(pipe_cv, param_grid, cv=3)
    grid_search.fit(x, y)
