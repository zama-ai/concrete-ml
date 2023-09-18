"""Tests for the sklearn models.

Generic tests test:
  - model (with n_bits)
  - FHE simulation or not
  - fit
  - double fit
  - compile
  - grid search
  - hyper parameters
  - offset
  - correctness (with accuracy and r2) of Concrete ML vs scikit-learn in clear
  - correctness tests with fhe = "disable", "simulate" and "execute", depending on
  limits (see N_BITS_THRESHOLD* constants) which are either due to execution time or limits of
  the compiler or minimal number of bits for precise computations
  - fit_benchmark
  - r2 score / accuracies
  - pandas
  - pipeline
  - calls to predict_proba
  - calls to decision_function

Are currently missing
  - check of predict_proba
  - check of decision_function

More information in https://github.com/zama-ai/concrete-ml-internal/issues/2682
"""

import copy

# pylint: disable=too-many-lines, too-many-arguments
import json
import tempfile
import warnings
from typing import Any, Dict, List

import numpy
import pandas
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.metrics import make_scorer, matthews_corrcoef, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from concrete.ml.common.serialization.dumpers import dump, dumps
from concrete.ml.common.serialization.loaders import load, loads
from concrete.ml.common.utils import (
    USE_OLD_VL,
    get_model_name,
    is_classifier_or_partial_classifier,
    is_model_class_in_a_list,
    is_regressor_or_partial_regressor,
)
from concrete.ml.pytest.utils import (
    _classifiers_and_datasets,
    instantiate_model_generic,
    sklearn_models_and_datasets,
)
from concrete.ml.sklearn import (
    get_sklearn_linear_models,
    get_sklearn_neighbors_models,
    get_sklearn_neural_net_models,
    get_sklearn_tree_models,
)

# Allow multiple runs in FHE to make sure we always have the correct output
N_ALLOWED_FHE_RUN = 5

# If n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS, we check correctness against
# scikit-learn in the clear, via check_correctness_with_sklearn function. This is because we need
# sufficiently number of bits for precision
N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS = 26

# We check correctness with check_is_good_execution_for_cml_vs_circuit or predict in
# fhe="disable" only if n_bits >= N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS. This is
# because we need sufficiently number of bits for precision
N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS = 6

# We never do checks with check_is_good_execution_for_cml_vs_circuit if
# n_bits >= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE. This is because computations are very
# slow
N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE = 17

assert (
    N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS <= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE
)

# If n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_EQUIVALENCE_TESTS, we check that the two models
# returned by fit_benchmark (the Concrete ML model and the scikit-learn model) are equivalent
N_BITS_THRESHOLD_FOR_SKLEARN_EQUIVALENCE_TESTS = 16

# There is a risk that no cryptographic parameters are available for high precision linear
# models. N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS makes sure we do not create linear models
# that do not have cryptographic parameters.
N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS = 11

# n_bits that we test, either in regular builds or just in weekly builds. 6 is to do tests in
# FHE which are not too long (relation with N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS and
# N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE). 26 is in relation with
# N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS, to do tests with check_correctness_with_sklearn
N_BITS_REGULAR_BUILDS = [6, 26]
N_BITS_WEEKLY_ONLY_BUILDS = [2, 8, 16]

# Circuit with 9 bits up to 16 bits are currently using the CRT circuit. We do not test them here
# as they take a bit more time than non-CRT based FHE circuit.
# N_BITS_THRESHOLD_FOR_CRT_FHE_CIRCUITS defines the threshold for which the circuit will be using
# the CRT.
N_BITS_THRESHOLD_FOR_CRT_FHE_CIRCUITS = 9


def get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option):
    """Prepare the the (x, y) data-set."""

    if not is_model_class_in_a_list(
        model_class, get_sklearn_linear_models() + get_sklearn_neighbors_models()
    ):
        if n_bits in N_BITS_WEEKLY_ONLY_BUILDS and not is_weekly_option:
            pytest.skip("Skipping some tests in non-weekly builds, except for linear models")

    # Get the data-set. The data generation is seeded in load_data.
    x, y = load_data(model_class, **parameters)

    return x, y


def preamble(model_class, parameters, n_bits, load_data, is_weekly_option):
    """Prepare the fitted model, and the (x, y) data-set."""

    if not is_model_class_in_a_list(
        model_class, get_sklearn_linear_models() + get_sklearn_neighbors_models()
    ):
        if n_bits in N_BITS_WEEKLY_ONLY_BUILDS and not is_weekly_option:
            pytest.skip("Skipping some tests in non-weekly builds")

    # Get the data-set. The data generation is seeded in load_data.
    model = instantiate_model_generic(model_class, n_bits=n_bits)
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    return model, x


def check_correctness_with_sklearn(
    model_class,
    x,
    y,
    n_bits,
    check_r2_score,
    check_accuracy,
    fhe="disable",
    hyper_parameters=None,
):
    """Check that Concrete ML and scikit-learn models are 'equivalent'."""

    if hyper_parameters is None:
        hyper_parameters = {}

    model = instantiate_model_generic(model_class, n_bits=n_bits, **hyper_parameters)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, sklearn_model = model.fit_benchmark(x, y)

    y_pred = model.predict(x)

    y_pred_sklearn = sklearn_model.predict(x)
    y_pred_cml = model.predict(x, fhe=fhe)

    # Check that the output shapes are correct
    assert y_pred.shape == y_pred_cml.shape, "Outputs have different shapes"

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
        "NeuralNetRegressor": -10,
    }

    # For classifiers
    threshold_accuracy_dic = {
        "LogisticRegression": 0.9,
        "LinearSVC": 0.9,
        "XGBClassifier": 0.7,
        "RandomForestClassifier": 0.8,
        "NeuralNetClassifier": 0.7,
        "KNeighborsClassifier": 0.9,
    }

    model_name = get_model_name(model_class)
    acceptance_r2score = acceptance_r2score_dic.get(model_name, 0.9)
    threshold_accuracy = threshold_accuracy_dic.get(model_name, 0.9)

    # If the model is a classifier, check that accuracies are similar
    if is_classifier_or_partial_classifier(model):
        check_accuracy(y_pred_sklearn, y_pred_cml, threshold=threshold_accuracy)

    # If the model is a regressor, check that R2 scores are similar
    else:
        assert is_regressor_or_partial_regressor(
            model
        ), "not a regressor, not a classifier, really?"
        check_r2_score(y_pred_sklearn, y_pred_cml, acceptance_score=acceptance_r2score)


def check_double_fit(model_class, n_bits, x_1, x_2, y_1, y_2):
    """Check double fit."""

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Set the torch seed manually before fitting a neural network
        if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):

            # Generate a seed for PyTorch
            main_seed = numpy.random.randint(0, 2**63)
            torch.manual_seed(main_seed)

        # Fit and predict on the first dataset
        model.fit(x_1, y_1)
        y_pred_1 = model.predict(x_1)

        # Store the input and output quantizers
        input_quantizers_1 = copy.copy(model.input_quantizers)
        output_quantizers_1 = copy.copy(model.output_quantizers)

        # Set the same torch seed manually before re-fitting the neural network
        if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
            torch.manual_seed(main_seed)

        # Re-fit on the second dataset
        model.fit(x_2, y_2)

        # Check that predictions are different
        y_pred_2 = model.predict(x_2)
        assert not numpy.array_equal(y_pred_1, y_pred_2)

        # Store the new input and output quantizers
        input_quantizers_2 = copy.copy(model.input_quantizers)
        output_quantizers_2 = copy.copy(model.output_quantizers)

        # Random forest and decision tree classifiers can have identical output_quantizers
        # This is because targets are integers, while these models have a fixed output
        # precision, which leads the output scale to be the same between models with similar target
        # classes range
        if is_model_class_in_a_list(
            model_class,
            get_sklearn_tree_models(
                classifier=True, str_in_class_name=["RandomForest", "DecisionTree"]
            ),
        ):
            quantizers_1 = input_quantizers_1
            quantizers_2 = input_quantizers_2
        else:
            quantizers_1 = input_quantizers_1 + output_quantizers_1
            quantizers_2 = input_quantizers_2 + output_quantizers_2

        # Check that the new quantizers are different from the first ones. This is because we
        # currently expect all quantizers to be re-computed when re-fitting a model

        assert all(
            quantizer_1 != quantizer_2
            for (quantizer_1, quantizer_2) in zip(quantizers_1, quantizers_2)
        )

        # Set the same torch seed manually before re-fitting the neural network
        if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
            torch.manual_seed(main_seed)

        # Re-fit on the first dataset again
        model.fit(x_1, y_1)

        # Check that predictions are identical to the first ones
        y_pred_3 = model.predict(x_1)
        assert numpy.array_equal(y_pred_1, y_pred_3)

        # Store the new input and output quantizers
        input_quantizers_3 = copy.copy(model.input_quantizers)
        output_quantizers_3 = copy.copy(model.output_quantizers)

        # Check that the new quantizers are identical from the first ones. Again, we expect the
        # quantizers to be re-computed when re-fitting. Since we used the same dataset as the first
        # fit, we also expect these quantizers to be the same.

        assert all(
            quantizer_1 == quantizer_3
            for (quantizer_1, quantizer_3) in zip(
                input_quantizers_1 + output_quantizers_1,
                input_quantizers_3 + output_quantizers_3,
            )
        )


def check_serialization(model, x, use_dump_method):
    """Check serialization."""

    check_serialization_dump_load(model, x, use_dump_method)
    check_serialization_dumps_loads(model, x, use_dump_method)


def check_serialization_dump_load(model, x, use_dump_method):
    """Check that a model can be serialized two times using dump/load."""

    with tempfile.TemporaryFile("w+") as temp_dump:
        # Dump the model into the file
        temp_dump.seek(0)
        temp_dump.truncate(0)
        if use_dump_method:
            model.dump(file=temp_dump)
        else:
            dump(model, file=temp_dump)

        # Load the model from the file as a dict using json
        temp_dump.seek(0)
        serialized_model_dict: Dict = json.load(temp_dump)

        # Load the model from the file using Concrete ML's method
        temp_dump.seek(0)
        loaded_model = load(file=temp_dump)

        # Dump the loaded model into the file using Concrete ML's method
        temp_dump.seek(0)
        temp_dump.truncate(0)
        if use_dump_method:
            loaded_model.dump(file=temp_dump)
        else:
            dump(loaded_model, file=temp_dump)

        # Load the model from the file again as a dict using json
        temp_dump.seek(0)
        re_serialized_model_dict: Dict = json.load(temp_dump)

        # Check that the dictionaries are identical
        # We exclude attributes such as `sklearn_model` (for linear and tree-based models) or
        # `params` (neural networks) since they are serialized using the pickle library, which does
        # not handle double serialization)
        for attribute in [
            "sklearn_model",
            "params",
            "criterion",
            "optimizer",
            "iterator_train",
            "iterator_valid",
            "dataset",
            "module__activation_function",
        ]:
            serialized_model_dict["serialized_value"].pop(attribute, None)
            re_serialized_model_dict["serialized_value"].pop(attribute, None)

        assert serialized_model_dict == re_serialized_model_dict

        # Check that the predictions made by both model are identical
        y_pred_model = model.predict(x)
        y_pred_loaded_model = loaded_model.predict(x)
        assert numpy.array_equal(y_pred_model, y_pred_loaded_model)

        # Check that the predictions made by both Scikit-Learn model are identical
        y_pred_sklearn_model = model.sklearn_model.predict(x)
        y_pred_loaded_sklearn_model = loaded_model.sklearn_model.predict(x)
        assert numpy.array_equal(y_pred_sklearn_model, y_pred_loaded_sklearn_model)


def check_serialization_dumps_loads(model, x, use_dump_method):
    """Check that a model can be serialized two times using dumps/loads."""

    # Dump the model as a string
    if use_dump_method:
        serialized_model_str = model.dumps()
    else:
        serialized_model_str = dumps(model)

    # Load the model from the string
    loaded_model = loads(serialized_model_str)

    # Dump the model as a string again
    if use_dump_method:
        re_serialized_model_str: str = loaded_model.dumps()
    else:
        re_serialized_model_str: str = dumps(loaded_model)  # type: ignore[no-redef]

    # Load both strings using json
    serialized_model_dict: Dict = json.loads(serialized_model_str)
    re_serialized_model_dict: Dict = json.loads(re_serialized_model_str)

    # Check that the dictionaries are identical
    # We exclude attributes such as `sklearn_model` (for linear and tree-based models) or
    # `params` (neural networks) since they are serialized using the pickle library, which does
    # not handle double serialization)
    for attribute in [
        "sklearn_model",
        "params",
        "criterion",
        "optimizer",
        "iterator_train",
        "iterator_valid",
        "dataset",
        "module__activation_function",
    ]:
        serialized_model_dict["serialized_value"].pop(attribute, None)
        re_serialized_model_dict["serialized_value"].pop(attribute, None)

    assert serialized_model_dict == re_serialized_model_dict

    # Check that the predictions made by both model are identical
    y_pred_model = model.predict(x)
    y_pred_loaded_model = loaded_model.predict(x)
    assert numpy.array_equal(y_pred_model, y_pred_loaded_model)

    # Check that the predictions made by both Scikit-Learn model are identical
    y_pred_sklearn_model = model.sklearn_model.predict(x)
    y_pred_loaded_sklearn_model = loaded_model.sklearn_model.predict(x)
    assert numpy.array_equal(y_pred_sklearn_model, y_pred_loaded_sklearn_model)


def check_offset(model_class, n_bits, x, y):
    """Check offset."""
    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Offsets are not supported by XGBoost
    if is_model_class_in_a_list(model_class, get_sklearn_tree_models(str_in_class_name="XGB")):
        # No pytest.skip, since it is not a bug but something which is inherent to XGB
        return

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Add the offset: here, we really need to fit, we can't reuse an already fitted model
        y += 3
        model.fit(x, y)
        model.predict(x[:1])

        # Another offset: here, we really need to fit, we can't reuse an already fitted model
        y -= 2
        model.fit(x, y)


def check_subfunctions(fitted_model, model_class, x):
    """Check subfunctions."""

    fitted_model.predict(x[:1])

    # skorch provides a predict_proba method for neural network regressors while Scikit-Learn does
    # not. We decided to follow Scikit-Learn's API as we build most of our tools on this library.
    # However, our models are still directly inheriting from skorch's classes, which makes this
    # method accessible by anyone, without having any FHE implementation. As this could create some
    # confusion, a NotImplementedError is raised. This issue could be fixed by making these classes
    # not inherit from skorch.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
    if get_model_name(fitted_model) == "NeuralNetRegressor":
        with pytest.raises(
            NotImplementedError,
            match=(
                "The `predict_proba` method is not implemented for neural network regressors. "
                "Please call `predict` instead."
            ),
        ):
            fitted_model.predict_proba(x)

    if get_model_name(fitted_model) == "KNeighborsClassifier":
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
        pytest.skip("Skipping subfunctions test for KNN, doesn't work for now")

    if is_classifier_or_partial_classifier(model_class):

        fitted_model.predict_proba(x)

        # Only linear classifiers have a decision function method
        if is_model_class_in_a_list(model_class, get_sklearn_linear_models()):
            fitted_model.decision_function(x)


def check_subfunctions_in_fhe(model, fhe_circuit, x):
    """Check subfunctions in FHE: calls and correctness."""

    # Generate the keys
    fhe_circuit.keygen()

    y_pred_fhe = []

    for _ in range(N_ALLOWED_FHE_RUN):
        for f_input in x:
            # Quantize an input (float)
            q_input = model.quantize_input(f_input.reshape(1, -1))

            # Encrypt the input
            q_input_enc = fhe_circuit.encrypt(q_input)

            # Execute the linear product in FHE
            q_y_enc = fhe_circuit.run(q_input_enc)

            # Decrypt the result (integer)
            q_y = fhe_circuit.decrypt(q_y_enc)

            # De-quantize the result
            y = model.dequantize_output(q_y)

            # Apply either the sigmoid if it is a binary classification task,
            # which is the case in this example, or a softmax function in order
            # to get the probabilities (in the clear)
            y_proba = model.post_processing(y)

            # Apply the argmax to get the class predictions (in the clear)
            if is_classifier_or_partial_classifier(model):
                y_class = numpy.argmax(y_proba, axis=-1)
                y_pred_fhe += list(y_class)
            else:
                y_pred_fhe += list(y_proba)

        # Compare with the FHE simulation mode
        y_pred_expected_in_simulation = model.predict(x, fhe="simulate")
        if numpy.isclose(numpy.array(y_pred_fhe), y_pred_expected_in_simulation).all():
            break

    assert numpy.isclose(numpy.array(y_pred_fhe), y_pred_expected_in_simulation).all(), (
        "computations are not the same between individual functions (in FHE) "
        "and predict function (in FHE simulation mode)"
    )


def check_input_support(model_class, n_bits, default_configuration, x, y, input_type):
    """Test all models with Pandas, List or Torch inputs."""

    def cast_input(x, y, input_type):
        "Convert x and y either in Pandas, List, Numpy or Torch type."

        assert input_type in ["pandas", "torch", "list", "numpy"], "Not a valid type casting"

        if input_type.lower() == "pandas":
            # Turn into Pandas
            x = pandas.DataFrame(x)
            y = pandas.Series(y) if y.ndim == 1 else pandas.DataFrame(y)
        elif input_type.lower() == "torch":
            # Turn into Torch
            x = torch.tensor(x)
            y = torch.tensor(y)
        elif input_type.lower() == "list":
            # Turn into List
            x = x.tolist()
            y = y.tolist()
        elif input_type.lower() == "numpy":
            assert isinstance(x, numpy.ndarray), f"Wrong type {type(x)}"
            assert isinstance(y, numpy.ndarray), f"Wrong type {type(y)}"
        return x, y

    model = instantiate_model_generic(model_class, n_bits=n_bits)
    x, y = cast_input(x, y, input_type=input_type)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    # Make sure `predict` is working when FHE is disabled
    model.predict(x)

    # Similarly, we test `predict_proba` for classifiers
    if is_classifier_or_partial_classifier(model):
        if get_model_name(model_class) == "KNeighborsClassifier":
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
            pytest.skip("Skipping predict_proba for KNN, doesn't work for now")
        model.predict_proba(x)

    # If n_bits is above N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS, do not compile the model
    # as there won't be any crypto parameters
    if n_bits >= N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS:
        return

    model.compile(x, default_configuration)

    # Make sure `predict` is working when FHE is disabled
    model.predict(x, fhe="simulate")

    # Similarly, we test `predict_proba` for classifiers
    if is_classifier_or_partial_classifier(model):
        model.predict_proba(x, fhe="simulate")


def check_pipeline(model_class, x, y):
    """Check pipeline support."""

    # Pipeline test sometimes fails with RandomForest models. This bug may come from Hummingbird
    # and needs further investigations
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2779
    if is_model_class_in_a_list(
        model_class, get_sklearn_tree_models(str_in_class_name="RandomForest")
    ):
        pytest.skip("Skipping pipeline test for RF, doesn't work for now")

    hyper_param_combinations = get_hyper_param_combinations(model_class)

    # Prepare the list of all hyper parameters
    hyperparameters_list = [
        {key: value} for key, values in hyper_param_combinations.items() for value in values
    ]

    # Take one of the hyper_parameters randomly (testing everything would be too long)
    if len(hyperparameters_list) == 0:
        hyper_parameters = {}
    else:
        hyper_parameters = hyperparameters_list[numpy.random.randint(0, len(hyperparameters_list))]

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2, random_state=numpy.random.randint(0, 2**15))),
            ("scaler", StandardScaler()),
            ("model", model_class(**hyper_parameters)),
        ]
    )

    # Do a grid search to find the best hyper-parameters
    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        param_grid = {
            "model__module__n_w_bits": [2, 3],
            "model__module__n_a_bits": [2, 3],
        }

    else:
        param_grid = {
            "model__n_bits": [2, 3],
        }
    # We need a small number of splits, especially for the KNN model, which has a small data-set
    grid_search = GridSearchCV(pipe_cv, param_grid, error_score="raise", cv=2)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        grid_search.fit(x, y)


def check_grid_search(model_class, x, y, scoring):
    """Check grid search."""
    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        param_grid = {
            "module__n_layers": [2, 3],
            "module__n_hidden_neurons_multiplier": [1],
            "module__activation_function": (nn.ReLU6,),
        }
    elif model_class in get_sklearn_tree_models(str_in_class_name="DecisionTree"):
        param_grid = {
            "n_bits": [20],
        }
    elif model_class in get_sklearn_tree_models():
        param_grid = {
            "n_bits": [20],
            "max_depth": [2],
            "n_estimators": [5, 10],
            "n_jobs": [1],
        }
    elif model_class in get_sklearn_neighbors_models():
        param_grid = {"n_bits": [2], "n_neighbors": [2]}
    else:
        param_grid = {
            "n_bits": [20],
        }

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        if get_model_name(model_class) == "KNeighborsClassifier" and scoring in [
            "roc_auc",
            "average_precision",
        ]:
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
            pytest.skip("Skipping predict_proba for KNN, doesn't work for now")

        _ = GridSearchCV(
            model_class(), param_grid, cv=2, scoring=scoring, error_score="raise", n_jobs=1
        ).fit(x, y)


def check_sklearn_equivalence(model_class, n_bits, x, y, check_accuracy, check_r2_score):
    """Check equivalence between the two models returned by fit_benchmark: the Concrete ML model and
    the scikit-learn model."""
    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # The `fit_benchmark` function of QNNs returns a QAT model and a FP32 model that is similar
    # in structure but trained from scratch. Furthermore, the `n_bits` setting
    # of the QNN instantiation in `instantiate_model_generic` takes `n_bits` as
    # a target accumulator and sets 3-b w&a for these tests. Thus it's
    # impossible to reach R-2 of 0.99 when comparing the two NN models returned by `fit_benchmark`
    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        pytest.skip("Skipping sklearn-equivalence test for NN, doesn't work for now")

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Random state should be taken from the method parameter
        model, sklearn_model = model.fit_benchmark(x, y)

    # If the model is a classifier
    if is_classifier_or_partial_classifier(model):

        # Check that accuracies are similar
        y_pred_cml = model.predict(x)
        y_pred_sklearn = sklearn_model.predict(x)
        check_accuracy(y_pred_sklearn, y_pred_cml)

        # If the model is a LinearSVC model, compute its predicted confidence score
        # This is done separately as scikit-learn doesn't provide a predict_proba method for
        # LinearSVC models
        if get_model_name(model_class) == "LinearSVC":
            y_pred_cml = model.decision_function(x)
            y_pred_sklearn = sklearn_model.decision_function(x)

        # Else, compute the model's predicted probabilities
        # predict_proba not implemented for KNeighborsClassifier for now
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
        elif get_model_name(model_class) != "KNeighborsClassifier":
            y_pred_cml = model.predict_proba(x)
            y_pred_sklearn = sklearn_model.predict_proba(x)

    # If the model is a regressor, compute its predictions
    else:
        y_pred_cml = model.predict(x)
        y_pred_sklearn = sklearn_model.predict(x)

    # Check that predictions, probabilities or confidence scores are similar using the R2 score
    check_r2_score(y_pred_sklearn, y_pred_cml)


def check_properties_of_circuit(model_class, fhe_circuit, check_circuit_has_no_tlu):
    """Check some properties of circuit, depending on the model class"""

    if is_model_class_in_a_list(model_class, get_sklearn_linear_models()):
        # Check that no TLUs are found within the MLIR
        check_circuit_has_no_tlu(fhe_circuit)


def get_hyper_param_combinations(model_class):
    """Return the hyper_param_combinations, depending on the model class"""
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
    elif model_class in get_sklearn_neighbors_models():
        # Use small `n_neighbors` values for KNN, because the data-set is too small for now
        hyper_param_combinations = {"n_neighbors": [1, 2]}
    else:

        assert is_model_class_in_a_list(
            model_class, get_sklearn_neural_net_models()
        ), "models are supposed to be tree-based or linear or QNN's"

        hyper_param_combinations = {}

    # Don't put n_bits in hyper_parameters, it comes from the test itself
    assert "n_bits" not in hyper_param_combinations

    return hyper_param_combinations


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
    hyper_param_combinations = get_hyper_param_combinations(model_class)

    # Prepare the list of all hyper parameters
    hyperparameters_list = [
        {key: value} for key, values in hyper_param_combinations.items() for value in values
    ]

    for hyper_parameters in hyperparameters_list:

        model = instantiate_model_generic(model_class, n_bits=n_bits, **hyper_parameters)

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2450
        # does not work for now, issue in HummingBird
        if get_model_name(model_class) == "RandomForestClassifier" and n_bits == 2:
            continue

        # Also fit with these hyper parameters to check it works fine
        with warnings.catch_warnings():
            # Sometimes, we miss convergence, which is not a problem for our test
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            # Here, we really need to fit, to take into account hyper parameters
            model.fit(x, y)

        # Check correctness with sklearn (if we have sufficiently bits of precision)
        if test_correctness_in_clear and n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS:
            check_correctness_with_sklearn(
                model_class,
                x,
                y,
                n_bits,
                check_r2_score,
                check_accuracy,
                fhe="disable",
                hyper_parameters=hyper_parameters,
            )


def check_fitted_compiled_error_raises(model_class, n_bits, x, y):
    """Check that methods that require the model to be compiled or fitted raise proper errors."""

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Quantizing inputs with an untrained model should not be possible
    with pytest.raises(AttributeError, match=".* model is not fitted.*"):
        model.quantize_input(x)

    # Quantizing outputs with an untrained model should not be possible
    with pytest.raises(AttributeError, match=".* model is not fitted.*"):
        model.dequantize_output(x)

    # Compiling an untrained model should not be possible
    with pytest.raises(AttributeError, match=".* model is not fitted.*"):
        model.compile(x)

    # Predicting in FHE using an untrained model should not be possible
    with pytest.raises(AttributeError, match=".* model is not fitted.*"):
        model.predict(x, fhe="execute")

    # Predicting in clear using an untrained model should not be possible for linear and
    # tree-based models
    if not is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        with pytest.raises(AttributeError, match=".* model is not fitted.*"):
            model.predict(x)

    if is_classifier_or_partial_classifier(model_class):
        if get_model_name(model) == "KNeighborsClassifier":
            pytest.skip("predict_proba not implement for KNN")
        # Predicting probabilities using an untrained linear or tree-based classifier should not
        # be possible
        if not is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
            with pytest.raises(AttributeError, match=".* model is not fitted.*"):
                model.predict_proba(x)

        # Predicting probabilities in FHE using an untrained QNN classifier should not be possible
        else:
            with pytest.raises(AttributeError, match=".* model is not fitted.*"):
                model.predict_proba(x, fhe="execute")

        # Computing the decision function using an untrained classifier should not be possible.
        # Note that the `decision_function` method is only available for linear models
        if is_model_class_in_a_list(model_class, get_sklearn_linear_models()):
            with pytest.raises(AttributeError, match=".* model is not fitted.*"):
                model.decision_function(x)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    # Predicting in FHE using a trained model that is not compiled should not be possible
    with pytest.raises(AttributeError, match=".* model is not compiled.*"):
        model.predict(x, fhe="execute")

    # Predicting probabilities in FHE using a trained QNN classifier that is not compiled should
    # not be possible
    if is_classifier_or_partial_classifier(model_class) and is_model_class_in_a_list(
        model_class, get_sklearn_neural_net_models()
    ):
        with pytest.raises(AttributeError, match=".* model is not compiled.*"):
            model.predict_proba(x, fhe="execute")


def check_class_mapping(model, x, y):
    """Check that classes with arbitrary labels are handled for all classifiers."""

    # Retrieve the data's target labels
    classes = numpy.unique(y)

    # Make sure these targets are ordered by default
    assert numpy.array_equal(numpy.arange(len(classes)), classes)

    # Fit the model
    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    # Compute the predictions
    y_pred = model.predict(x)

    # Shuffle the initial labels (in place)
    numpy.random.shuffle(classes)

    # Map each targets' label to the the new shuffled ones
    new_y = classes[y]

    # Fit the model using these new targets
    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, new_y)

    # Compute the predictions
    y_pred_shuffled = model.predict(x)

    # Check that the mapping of labels was kept by Concrete ML
    numpy.array_equal(classes[y_pred], y_pred_shuffled)


def check_exposition_of_sklearn_attributes(model, x, y):
    """Check training scikit-learn attributes are properly exposed in our models."""

    training_attribute = "coef_"
    # Check that accessing an attribute that follows scikit-learn's naming convention for training
    # attributes by ending with an underscore properly raises an Attribute error when the model is
    # not fitted
    with pytest.raises(
        AttributeError,
        match=f".* {training_attribute} cannot be found in the Concrete ML.*",
    ):
        getattr(model, training_attribute)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    for name in vars(model.sklearn_model):
        if name.endswith("_") and not name.endswith("__"):
            assert hasattr(
                model, name
            ), f"Training attribute {name} is not exposed in {get_model_name(model)} model."

    wrong_training_attribute_1 = "concrete_ml"
    # Check that accessing an unknown attribute properly raises an Attribute error
    with pytest.raises(
        AttributeError,
        match=f".* {wrong_training_attribute_1} cannot be found in the Concrete ML.*",
    ):
        getattr(model, wrong_training_attribute_1)

    wrong_training_attribute_2 = "concrete_ml_"
    # Check that accessing an unknown attribute that almost follows scikit-learn's naming
    # convention for training attributes by ending with two underscores properly raises an
    # Attribute error
    with pytest.raises(
        AttributeError,
        match=f".* object has no attribute '{wrong_training_attribute_2}'",
    ):
        getattr(model, wrong_training_attribute_2)

    wrong_training_attribute_3 = "concrete_ml__"
    # Check that accessing an unknown attribute that almost follows scikit-learn's naming
    # convention for training attributes by ending with two underscores properly raises an
    # Attribute error
    with pytest.raises(
        AttributeError,
        match=f".* {wrong_training_attribute_3} cannot be found in the Concrete ML.*",
    ):
        getattr(model, wrong_training_attribute_3)


def check_exposition_structural_methods_decision_trees(model, x, y):
    """Check structural methods from scikit-learn are properly exposed in decision tree models."""

    # Check that accessing an attribute that follows scikit-learn's naming convention for training
    # attributes by ending with an underscore properly raises an Attribute error when the model is
    # not fitted
    with pytest.raises(
        AttributeError,
        match=".* get_n_leaves cannot be found in the Concrete ML.*",
    ):
        model.get_n_leaves()

    with pytest.raises(
        AttributeError,
        match=".* get_depth cannot be found in the Concrete ML.*",
    ):
        model.get_depth()

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    # Get the number of leaves from both the scikit-learn and Concrete ML models
    concrete_value = model.get_n_leaves()
    sklearn_value = model.sklearn_model.get_n_leaves()

    model_name = get_model_name(model)

    assert concrete_value == sklearn_value, (
        f"Method get_n_leaves of model {model_name} do not output the same value as with its "
        f"scikit-learn equivalent. Got {concrete_value}, expected {sklearn_value}."
    )

    # Get the tree depth from both the scikit-learn and Concrete ML models
    concrete_value = model.get_depth()
    sklearn_value = model.sklearn_model.get_depth()

    model_name = get_model_name(model)

    assert concrete_value == sklearn_value, (
        f"Method get_depth of model {model_name} do not output the same value as with its "
        f"scikit-learn equivalent. Got {concrete_value}, expected {sklearn_value}."
    )


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    [
        n
        for n in N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS
        if n >= N_BITS_THRESHOLD_FOR_SKLEARN_EQUIVALENCE_TESTS
    ],
)
def test_quantization(
    model_class,
    parameters,
    n_bits,
    load_data,
    check_r2_score,
    check_accuracy,
    is_weekly_option,
    verbose=True,
):
    """Test quantization."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_sklearn_equivalence")

    check_sklearn_equivalence(model_class, n_bits, x, y, check_accuracy, check_r2_score)


# This test is a known flaky
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3661
@pytest.mark.flaky
@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    [
        n
        for n in N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS
        if n >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS
    ],
)
def test_correctness_with_sklearn(
    model_class,
    parameters,
    n_bits,
    load_data,
    check_r2_score,
    check_accuracy,
    is_weekly_option,
    verbose=True,
):
    """Test that Concrete ML and scikit-learn models are 'equivalent'."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Check correctness with sklearn (if we have sufficiently bits of precision)
    if verbose:
        print("Run check_correctness_with_sklearn with fhe='disable'")

    check_correctness_with_sklearn(
        model_class,
        x,
        y,
        n_bits,
        check_r2_score,
        check_accuracy,
        fhe="disable",
    )


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS,
)
def test_hyper_parameters(
    model_class,
    parameters,
    n_bits,
    load_data,
    check_r2_score,
    check_accuracy,
    is_weekly_option,
    verbose=True,
):
    """Testing hyper parameters."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_hyper_parameters")

    test_correctness_in_clear = True

    check_hyper_parameters(
        model_class,
        n_bits,
        x,
        y,
        test_correctness_in_clear,
        check_r2_score,
        check_accuracy,
    )


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize("n_bits", [3])
# The complete list of built-in scoring functions can be found in scikit-learn's documentation:
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Here, we only consider the main ones
@pytest.mark.parametrize(
    "scoring, is_classification",
    [
        pytest.param("accuracy", True),
        pytest.param("balanced_accuracy", True),
        pytest.param(make_scorer(top_k_accuracy_score, k=1), True, id="top_k_accuracy"),
        pytest.param("average_precision", True),
        pytest.param("f1", True),
        pytest.param("precision", True),
        pytest.param("recall", True),
        pytest.param("roc_auc", True),
        pytest.param(
            make_scorer(matthews_corrcoef, greater_is_better=True), True, id="matthews_corrcoef"
        ),
        pytest.param("explained_variance", False),
        pytest.param("max_error", False),
        pytest.param("neg_mean_absolute_error", False),
        pytest.param("neg_mean_squared_error", False),
        pytest.param("neg_root_mean_squared_error", False),
        pytest.param("neg_median_absolute_error", False),
        pytest.param("r2", False),
        pytest.param("neg_mean_absolute_percentage_error", False),
    ],
)
def test_grid_search(
    model_class,
    parameters,
    n_bits,
    scoring,
    is_classification,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test Grid search."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    # If the scoring function is meant for classifiers (resp. regressors) but the model is not a
    # classifier (resp. regressor), skip the test
    # Also skip the test if the classification data is multi-class as most of the scoring functions
    # tested here don't support that
    if is_classification:
        if (
            not is_classifier_or_partial_classifier(model_class)
            or parameters.get("n_classes", 2) > 2
        ):
            return
    elif not is_regressor_or_partial_regressor(model_class):
        return

    # Max error does not support multi-output models
    if scoring == "max_error" and parameters.get("n_targets", 1) > 1:
        return

    if verbose:
        print("Run check_grid_search")

    check_grid_search(model_class, x, y, scoring)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize("use_dump_method", [True, False])
def test_serialization(
    model_class,
    parameters,
    use_dump_method,
    load_data,
    is_weekly_option,
    default_configuration,
    verbose=True,
):
    """Test Serialization."""
    # This test only checks the serialization's functionalities, so there is no need to test it
    # over several n_bits
    n_bits = min(N_BITS_REGULAR_BUILDS)

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Compile the model to make sure we consider all possible attributes during the serialization
    model.compile(x, default_configuration)

    if verbose:
        print("Run check_serialization")

    check_serialization(model, x, use_dump_method)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS,
)
def test_double_fit(
    model_class,
    parameters,
    n_bits,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test Double fit."""

    # Generate a random state for generating the first dataset
    random_state = numpy.random.randint(0, 2**15)
    parameters["random_state"] = random_state

    # Generate two different datasets
    x_1, y_1 = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Make sure the second dataset is different by using a distinct random state
    parameters["random_state"] += 1
    x_2, y_2 = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_double_fit")

    check_double_fit(model_class, n_bits, x_1, x_2, y_1, y_2)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS,
)
def test_offset(
    model_class,
    parameters,
    n_bits,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test with offset."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_offset")

    check_offset(model_class, n_bits, x, y)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS,
)
@pytest.mark.parametrize("input_type", ["numpy", "torch", "pandas", "list"])
def test_input_support(
    model_class,
    parameters,
    n_bits,
    load_data,
    input_type,
    default_configuration,
    is_weekly_option,
    verbose=True,
):
    """Test all models with Pandas, List or Torch inputs."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run input_support")

    check_input_support(model_class, n_bits, default_configuration, x, y, input_type)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS,
)
def test_subfunctions(
    model_class,
    parameters,
    n_bits,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test subfunctions."""
    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_subfunctions")

    check_subfunctions(model, model_class, x)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "n_bits",
    N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS,
)
def test_pipeline(
    model_class,
    parameters,
    n_bits,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test with pipelines."""
    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_pipeline")

    check_pipeline(model_class, x, y)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "simulate",
    [
        pytest.param(False, id="fhe"),
        pytest.param(True, id="simulate"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        n
        for n in N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS
        if n >= N_BITS_THRESHOLD_FOR_PREDICT_CORRECTNESS_TESTS
    ],
)
# pylint: disable=too-many-branches
def test_predict_correctness(
    model_class,
    parameters,
    simulate,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
    check_circuit_has_no_tlu,
    is_weekly_option,
    test_subfunctions_in_fhe=True,
    verbose=True,
):
    """Test correct execution, if there is sufficiently n_bits."""

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    # How many samples for tests in FHE (i.e., predict with fhe = "execute" or "simulate")
    if is_weekly_option or simulate:
        number_of_tests_in_fhe = 5
    else:
        number_of_tests_in_fhe = 1

    # How many samples for tests in quantized module (i.e., predict with fhe = "disable")
    if is_weekly_option:
        number_of_tests_in_non_fhe = 50
    else:
        number_of_tests_in_non_fhe = 10

    # Do some inferences in clear
    if verbose:
        print(
            "Inference in the clear (with "
            f"number_of_tests_in_non_fhe = {number_of_tests_in_non_fhe})"
        )
    # KNN works only for smaller quantization bits
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3979
    if n_bits > 5 and get_model_name(model) == "KNeighborsClassifier":
        pytest.skip("Use less than 5 bits with KNN.")

    y_pred = model.predict(x[:number_of_tests_in_non_fhe])

    list_of_possibilities = [False, True]

    # Prevent computations in FHE if too many bits
    if n_bits >= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE:
        list_of_possibilities = [False]

    for test_with_execute_in_fhe in list_of_possibilities:

        # N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS bits is currently the
        # limit to find crypto parameters for linear models
        # make sure we only compile below that bit-width.
        if test_with_execute_in_fhe and not n_bits >= N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS:

            if verbose:
                print("Compile the model")

            with warnings.catch_warnings():
                fhe_circuit = model.compile(
                    x,
                    default_configuration,
                    show_mlir=verbose and (n_bits <= 8),
                )

                check_properties_of_circuit(model_class, fhe_circuit, check_circuit_has_no_tlu)

            if verbose:
                print("Compilation done")

            if verbose:
                print(
                    "Run check_is_good_execution_for_cml_vs_circuit "
                    + f"(with number_of_tests_in_fhe = {number_of_tests_in_fhe})"
                )

            # Check the `predict` method
            check_is_good_execution_for_cml_vs_circuit(
                x[:number_of_tests_in_fhe], model=model, simulate=simulate
            )

            if test_subfunctions_in_fhe and (not simulate):
                if verbose:
                    print("Testing subfunctions in FHE")

                check_subfunctions_in_fhe(model, fhe_circuit, x[:number_of_tests_in_fhe])

        else:
            if verbose:
                print(
                    "Run predict in fhe='disable' "
                    f"(with number_of_tests_in_non_fhe = {number_of_tests_in_non_fhe})"
                )

            # At least, check in clear mode
            y_pred_fhe = model.predict(x[:number_of_tests_in_non_fhe], fhe="disable")

            # Check that the output shape is correct
            assert y_pred_fhe.shape == y_pred.shape
            assert numpy.array_equal(y_pred_fhe, y_pred)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
def test_fitted_compiled_error_raises(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test Fit and Compile error raises."""
    n_bits = min(N_BITS_REGULAR_BUILDS)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_fitted_compiled_error_raises")

    check_fitted_compiled_error_raises(model_class, n_bits, x, y)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize(
    "error_param",
    [{"p_error": 0.9999999999990905}],  # 1 - 2**-40
    ids=["p_error"],
)
def test_p_error_global_p_error_simulation(
    model_class,
    parameters,
    error_param,
    load_data,
    is_weekly_option,
):
    """Test p_error and global_p_error simulation.

    Description:
        A model is compiled with a large p_error. The test then checks the predictions for
        simulated and fully homomorphic encryption (FHE) inference, and asserts
        that the predictions for both are different from the expected predictions.
    """
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3297
    if "global_p_error" in error_param:
        pytest.skip("global_p_error behave very differently depending on the type of model.")

    if get_model_name(model_class) == "KNeighborsClassifier":
        # KNN works only for smaller quantization bits
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3979
        n_bits = min([2] + N_BITS_REGULAR_BUILDS)
    else:
        n_bits = min(N_BITS_REGULAR_BUILDS)

    # Get data-set, initialize and fit the model
    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Check if model is linear
    is_linear_model = is_model_class_in_a_list(model_class, get_sklearn_linear_models())

    # Compile with a large p_error to be sure the result is random.
    model.compile(x, **error_param)

    def check_for_divergent_predictions(x, model, fhe, max_iterations=N_ALLOWED_FHE_RUN):
        """Detect divergence between simulated/FHE execution and clear run."""
        predict_function = (
            model.predict_proba
            if is_classifier_or_partial_classifier(model)
            # `predict_prob` not implemented yet for KNeighborsClassifier
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
            and get_model_name(model) != "KNeighborsClassifier"
            else model.predict
        )
        y_expected = predict_function(x, fhe="disable")
        for i in range(max_iterations):
            y_pred = predict_function(x[i : i + 1], fhe=fhe).ravel()
            if not numpy.array_equal(y_pred, y_expected[i : i + 1].ravel()):
                return True
        return False

    simulation_diff_found = check_for_divergent_predictions(x, model, fhe="simulate")
    fhe_diff_found = check_for_divergent_predictions(x, model, fhe="execute")

    # Check for differences in predictions
    # Remark that, with the old VL, linear models (or, more generally, circuits without PBS) were
    # badly simulated. It has been fixed in the new simulation.
    if is_linear_model and USE_OLD_VL:

        # In FHE, high p_error affect the crypto parameters which
        # makes the predictions slightly different
        assert fhe_diff_found, "FHE predictions should be different for linear models"

        # linear models p_error is not simulated
        assert not simulation_diff_found, "SIMULATE predictions not the same for linear models"

    else:
        assert fhe_diff_found and simulation_diff_found, (
            f"Predictions not different in at least one run.\n"
            f"FHE predictions differ: {fhe_diff_found}\n"
            f"SIMULATE predictions differ: {simulation_diff_found}"
        )


@pytest.mark.parametrize("model_class, parameters", _classifiers_and_datasets)
def test_class_mapping(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test class mapping for classifiers."""
    n_bits = min(N_BITS_REGULAR_BUILDS)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    if verbose:
        print("Run check_class_mapping")

    check_class_mapping(model, x, y)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
def test_exposition_of_sklearn_attributes(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test the exposition of scikit-learn training attributes."""
    n_bits = min(N_BITS_REGULAR_BUILDS)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    if verbose:
        print("Run check_exposition_of_sklearn_attributes")

    check_exposition_of_sklearn_attributes(model, x, y)


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
def test_exposition_structural_methods_decision_trees(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test the exposition of specific structural methods found in decision tree models."""
    if get_model_name(model_class) not in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
        return

    n_bits = min(N_BITS_REGULAR_BUILDS)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    if verbose:
        print("Run check_exposition_structural_methods_decision_trees")

    check_exposition_structural_methods_decision_trees(model, x, y)
