"""Tests for the sklearn decision trees."""
import warnings
from typing import Any, Dict, List

import numpy
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import GridSearchCV
from torch import nn

from concrete.ml.common.utils import get_model_name, is_model_class_in_a_list
from concrete.ml.pytest.utils import classifiers, regressors
from concrete.ml.sklearn.base import (
    get_sklearn_linear_models,
    get_sklearn_neural_net_models,
    get_sklearn_tree_models,
)

# 0. If n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS, we check correctness against
# scikit-learn in the clear, via check_correctness_with_sklearn function. This is because we need
# sufficiently number of bits for precision
N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS = 16

# 1. We force the use of the VL if n_bits > N_BITS_THRESHOLD_TO_FORCE_VL. This is because compiler
# limits to 16b in FHE
N_BITS_THRESHOLD_TO_FORCE_VL = 16

# 2. We check correctness with check_is_good_execution_for_cml_vs_circuit or predict in
# execute_in_fhe=False only if n_bits >= N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS. This is
# because we need sufficiently number of bits for precision
N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS = 6

# 3. We never do checks with check_is_good_execution_for_cml_vs_circuit if
# n_bits >= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE. This is because computations are very
# slow
N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE = 7

assert (
    N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS <= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE
)

# 4. How many samples for tests in FHE (ie, predict with execute_in_fhe = True)
NUMBER_OF_TESTS_IN_FHE = 5

# 5. How many samples for tests in quantized module (ie, predict with execute_in_fhe = False)
NUMBER_OF_TESTS_IN_NON_FHE = 50


# pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements
def check_generic(
    model_class,
    parameters,
    use_virtual_lib,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
    check_r2_score,
    check_accuracy,
    test_hyper_parameters=True,
    test_grid_search=True,
    test_double_fit=True,
    test_offset=True,
    test_correctness_in_clear=True,
    verbose=True,
):
    """Generic tests.

    This function tests:
      - model (with n_bits)
      - virtual_lib or not
      - fit
      - double fit
      - compile
      - grid search
      - hyper parameters
      - offset
      - correctness (with accuracy and r2) of Concrete-ML vs scikit-learn in clear
      - correctness tests with or without VL, with execute_in_fhe = True and False, depending on
      limits (see N_BITS_THRESHOLD* constants) which are either due to execution time or limits of
      the compiler or minimal number of bits for precise computations
      - fit_benchmark
      - r2 score / accuracies

    Are currently missing
      - quantization
      - pandas
      - pipeline

    More information in https://github.com/zama-ai/concrete-ml-internal/issues/2682

    """

    if verbose:
        print("Generic tests")

    assert (
        n_bits <= N_BITS_THRESHOLD_TO_FORCE_VL or use_virtual_lib
    ), "VL should be used for larger precisions"

    # Get the dataset. The data generation is seeded in load_data.
    model_name = get_model_name(model_class)
    x, y = load_data(**parameters, model_name=model_name)

    # Set the model
    model = model_class(n_bits=n_bits)

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    if verbose:
        print("Fit")

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    if verbose:
        print("Inference in the clear")

    if verbose:
        print(f"Compile {use_virtual_lib=}")

    with warnings.catch_warnings():
        # Use virtual lib to not have issues with precision
        model.compile(
            x,
            default_configuration,
            use_virtual_lib=use_virtual_lib,
            show_mlir=verbose,
        )

    if verbose:
        print("Compilation done")

    # Check correctness with sklearn (if we have sufficiently bits of precision)
    if test_correctness_in_clear and n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS:
        if verbose:
            print("Run check_correctness_with_sklearn with execute_in_fhe=False")

        check_correctness_with_sklearn(
            model_class,
            x,
            y,
            check_r2_score,
            check_accuracy,
            hyper_parameters_including_n_bits={"n_bits": n_bits},
            execute_in_fhe=False,
        )

    # Testing hyper parameters
    if test_hyper_parameters:
        if verbose:
            print("Run check_hyper_parameters")

        check_hyper_parameters(
            model_class,
            n_bits,
            x,
            y,
            test_correctness_in_clear,
            check_r2_score,
            check_accuracy,
        )

    # Grid search
    if test_grid_search:
        if verbose:
            print("Run check_grid_search")

        check_grid_search(model, model_name, model_class, x, y)

    # Double fit
    if test_double_fit:
        if verbose:
            print("Run check_double_fit")

        check_double_fit(model_class, n_bits, x, y)

    # Test with offset
    if test_offset:
        if verbose:
            print("Run check_offset")

        check_offset(model_class, n_bits, x, y)

    # Do some inferences in clear
    y_pred = model.predict(x[:NUMBER_OF_TESTS_IN_NON_FHE])

    # Check correct execution, if there is sufficiently n_bits
    if n_bits >= N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS:

        for test_with_execute_in_fhe in [False, True]:

            # Prevent computations in FHE if too many bits
            if n_bits >= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE:
                test_with_execute_in_fhe = False

            if test_with_execute_in_fhe:
                if verbose:
                    print("Run check_is_good_execution_for_cml_vs_circuit")

                check_is_good_execution_for_cml_vs_circuit(
                    x[:NUMBER_OF_TESTS_IN_FHE], model_function=model
                )
            else:
                if verbose:
                    print("Run predict in execute_in_fhe=False")

                # At least, check without execute_in_fhe
                y_pred_fhe = model.predict(x[:NUMBER_OF_TESTS_IN_NON_FHE], execute_in_fhe=False)

                # Check that the output shape is correct
                assert y_pred_fhe.shape == y_pred.shape
                assert numpy.array_equal(y_pred_fhe, y_pred)


def check_correctness_with_sklearn(
    model_class,
    x,
    y,
    check_r2_score,
    check_accuracy,
    hyper_parameters_including_n_bits,
    execute_in_fhe=False,
):
    """Check that Concrete-ML and scikit-learn models are 'equivalent'."""
    assert "n_bits" in hyper_parameters_including_n_bits

    model_name = get_model_name(model_class)
    model = model_class(**hyper_parameters_including_n_bits)

    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        x = x.astype(numpy.float32)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, sklearn_model = model.fit_benchmark(x, y)

    y_pred = model.predict(x)

    y_pred_sklearn = sklearn_model.predict(x)
    y_pred_cml = model.predict(x, execute_in_fhe=execute_in_fhe)

    # Check that the output shapes are correct
    assert y_pred.shape == y_pred_cml.shape, "Quantized clear and FHE outputs have different shapes"

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2604
    # Generic tests look to show issues in accuracy / R2 score, even for high n_bits

    # For regressions
    acceptance_r2score_dic = {
        "TweedieRegressor": 0.9,
        "GammaRegressor": 0.9,
        "LinearRegression": 0.9,
        "LinearSVR": 0.9,
        "PoissonRegressor": 0.9,
        "Lasso": 0.9,
        "Ridge": 0.9,
        "ElasticNet": 0.9,
        "XGBRegressor": -0.2,
        "NeuralNetRegressor": -3,
    }

    # For classifiers
    threshold_accuracy_dic = {
        "LogisticRegression": 0.9,
        "LinearSVC": 0.9,
        "XGBClassifier": 0.7,
        "RandomForestClassifier": 0.8,
        "NeuralNetClassifier": 0.7,
    }

    acceptance_r2score = acceptance_r2score_dic.get(model_name, 0.9)
    threshold_accuracy = threshold_accuracy_dic.get(model_name, 0.9)

    # If the model is a classifier, check that accuracies are similar
    if is_classifier(model):
        check_accuracy(y_pred_sklearn, y_pred_cml, threshold=threshold_accuracy)

    # If the model is a regressor, check that R2 scores are similar
    else:
        assert is_regressor(model), "not a regressor, not a classifier, really?"
        check_r2_score(y_pred_sklearn, y_pred_cml, acceptance_score=acceptance_r2score)


def check_double_fit(model_class, n_bits, x, y):
    """Check double fit."""
    model_name = get_model_name(model_class)

    model = model_class(n_bits=n_bits)

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    if "NeuralNet" in model_name:
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/918
        # This fails and needs to be fixed
        return

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # First fit
        model.fit(x, y)
        y_pred_one = model.predict(x)

        # Second fit
        model.fit(x, y)
        y_pred_two = model.predict(x)

    assert numpy.array_equal(y_pred_one, y_pred_two)


def check_offset(model_class, n_bits, x, y):
    """Check offset."""
    model_name = get_model_name(model_class)

    if "XGB" in model_name:
        # Offsets are not supported by XGBoost
        return

    if "NeuralNet" in model_name:
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2404
        # Check it is not supported
        return

    model = model_class(n_bits=n_bits)

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Add the offset
        y += 3
        model.fit(x, y)
        model.predict(x[:1])

        # Another offset
        y -= 2
        model.fit(x, y)


def check_grid_search(model, model_name, model_class, x, y):
    """Check grid search."""
    if is_classifier(model_class):
        grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    else:
        grid_scorer = make_scorer(mean_squared_error, greater_is_better=True)

    if "NeuralNet" in model_name:
        param_grid = {
            "module__n_layers": (3, 5),
            "module__activation_function": (nn.Tanh, nn.ReLU6),
        }
    elif model_class in get_sklearn_tree_models(str_in_class_name="DecisionTree"):
        param_grid = {
            "n_bits": [20],
        }
    elif model_class in get_sklearn_tree_models():
        param_grid = {
            "n_bits": [20],
            "max_depth": [2],
            "n_estimators": [5, 10, 50, 100],
            "n_jobs": [1],
        }
    else:
        param_grid = {
            "n_bits": [20],
        }

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        _ = GridSearchCV(
            model, param_grid, cv=5, scoring=grid_scorer, error_score="raise", n_jobs=1
        ).fit(x, y)


def check_hyper_parameters(
    model_class,
    n_bits,
    x,
    y,
    test_correctness_in_clear,
    check_r2_score,
    check_accuracy,
):
    """Check hyper parameters."""
    hyper_param_combinations: Dict[str, List[Any]]

    if is_model_class_in_a_list(model_class, get_sklearn_linear_models()):
        hyper_param_combinations = {"fit_intercept": [False, True]}
    elif model_class in get_sklearn_tree_models(str_in_class_name="DecisionTree"):
        hyper_param_combinations = {}
    elif model_class in get_sklearn_tree_models(str_in_class_name="RandomForest"):
        hyper_param_combinations = {
            "max_depth": [3, 4, 5, 10],
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 3, 4],
            "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3],
            "max_features": ["sqrt", "log2"],
            "max_leaf_nodes": [None, 5, 10, 20],
        }
    elif model_class in get_sklearn_tree_models(str_in_class_name="XGB"):
        hyper_param_combinations = {
            "max_depth": [3, 4, 5, 10],
            "learning_rate": [1, 0.5, 0.1],
            "n_estimators": [1, 50, 100, 1000],
            "tree_method": ["auto", "exact", "approx"],
            "gamma": [0, 0.1, 0.5],
            "min_child_weight": [1, 5, 10],
            "max_delta_step": [0, 0.5, 0.7],
            "subsample": [0.5, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.9, 1.0],
            "colsample_bylevel": [0.5, 0.9, 1.0],
            "colsample_bynode": [0.5, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "scale_pos_weight": [0.5, 0.9, 1.0],
            "importance_type": ["weight", "gain"],
            "base_score": [0.5, None],
        }
    else:

        assert is_model_class_in_a_list(
            model_class, get_sklearn_neural_net_models()
        ), "models are supposed to be tree-based or linear or QNN's"

        hyper_param_combinations = {}

    # Don't put n_bits in hyper_parameters, it comes from the test itself
    assert "n_bits" not in hyper_param_combinations

    # Prepare the list of all hyper parameters
    hyperparameters_list = [
        {key: value} for key, values in hyper_param_combinations.items() for value in values
    ]

    for hyper_parameters in hyperparameters_list:

        # Add n_bits
        hyper_parameters["n_bits"] = n_bits

        model = model_class(**hyper_parameters)

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2450
        # does not work for now, issue in HummingBird
        model_name = get_model_name(model_class)
        if model_name == "RandomForestClassifier" and hyper_parameters["n_bits"] == 2:
            continue

        # Also fit with these hyper parameters to check it works fine
        with warnings.catch_warnings():
            # Sometimes, we miss convergence, which is not a problem for our test
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x, y)

        # Check correctness with sklearn (if we have sufficiently bits of precision)
        if test_correctness_in_clear and n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS:
            check_correctness_with_sklearn(
                model_class,
                x,
                y,
                check_r2_score,
                check_accuracy,
                hyper_parameters_including_n_bits=hyper_parameters,
                execute_in_fhe=False,
            )


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
@pytest.mark.parametrize(
    "use_virtual_lib",
    [
        pytest.param(False, id="no_virtual_lib"),
        pytest.param(True, id="use_virtual_lib"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2761
        # restore these tests when compilation time for 16b program has been reduced
        #
        # Make it back when compilation is fast pytest.param(16, id="n_bits_16"),
        # Make it back when compilation is fast pytest.param(8, id="n_bits_8"),
        pytest.param(6, id="n_bits_6"),
        pytest.param(2, id="n_bits_2"),
    ],
)
def test_generic(
    model,
    parameters,
    use_virtual_lib,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
    check_r2_score,
    check_accuracy,
):
    """Test in a generic way."""

    if n_bits > N_BITS_THRESHOLD_TO_FORCE_VL and not use_virtual_lib:
        # Skip this, we can't use FHE for too large precision
        pytest.skip("We can't use FHE for too large precision")

    check_generic(
        model,
        parameters,
        use_virtual_lib,
        n_bits,
        load_data,
        default_configuration,
        check_is_good_execution_for_cml_vs_circuit,
        check_r2_score,
        check_accuracy,
    )
