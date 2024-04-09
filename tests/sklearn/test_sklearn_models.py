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
"""

import copy
import json
import os
import sys
import tempfile

# pylint: disable=too-many-lines, too-many-arguments
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
    array_allclose_and_same_shape,
    get_model_class,
    get_model_name,
    is_classifier_or_partial_classifier,
    is_model_class_in_a_list,
    is_regressor_or_partial_regressor,
)
from concrete.ml.pytest.utils import (
    MODELS_AND_DATASETS,
    UNIQUE_MODELS_AND_DATASETS,
    get_random_samples,
    get_sklearn_all_models_and_datasets,
    get_sklearn_linear_models_and_datasets,
    get_sklearn_neighbors_models_and_datasets,
    get_sklearn_tree_models_and_datasets,
    instantiate_model_generic,
)
from concrete.ml.sklearn import (
    _get_sklearn_linear_models,
    _get_sklearn_neighbors_models,
    _get_sklearn_neural_net_models,
    _get_sklearn_tree_models,
)

# Allow multiple runs in FHE to make sure we always have the correct output
N_ALLOWED_FHE_RUN = 5

# If n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS, we check correctness against
# scikit-learn in the clear, via check_correctness_with_sklearn function. This is because we need
# sufficiently number of bits for precision
N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS = 26

# We never do checks with check_is_good_execution_for_cml_vs_circuit if
# n_bits >= N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE. This is because computations are very
# slow
N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE = 17

# If n_bits >= N_BITS_THRESHOLD_FOR_SKLEARN_EQUIVALENCE_TESTS, we check that the two models
# returned by fit_benchmark (the Concrete ML model and the scikit-learn model) are equivalent
N_BITS_THRESHOLD_FOR_SKLEARN_EQUIVALENCE_TESTS = 16

# There is a risk that no cryptographic parameters are available for high precision linear
# models. N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS makes sure we do not create linear models
# that do not have cryptographic parameters.
N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS = 11

# n_bits that we test, either in regular builds or just in weekly builds. 6 is to do tests in
# FHE which are not too long (relation with N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE).
# 26 is in relation with N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS, to do tests with
# check_correctness_with_sklearn
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
        model_class, _get_sklearn_linear_models() + _get_sklearn_neighbors_models()
    ):
        if n_bits in N_BITS_WEEKLY_ONLY_BUILDS and not is_weekly_option:
            pytest.skip("Skipping some tests in non-weekly builds")

    # Get the data-set. The data generation is seeded in load_data.
    x, y = load_data(model_class, **parameters)

    return x, y


def preamble(model_class, parameters, n_bits, load_data, is_weekly_option):
    """Prepare the fitted model, and the (x, y) data-set."""

    if not is_model_class_in_a_list(
        model_class, _get_sklearn_linear_models() + _get_sklearn_neighbors_models()
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


def get_n_bits_non_correctness(model_class):
    """Get the number of bits to use for non correctness related tests."""

    # KNN can only be compiled with small quantization bit numbers for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3979
    if get_model_name(model_class) == "KNeighborsClassifier":
        n_bits = 2
    else:
        n_bits = min(N_BITS_REGULAR_BUILDS)

    return n_bits


def fit_and_compile(model, x, y):
    """Fit the model and compile it."""

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    model.compile(x)


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

    model_name = get_model_name(model_class)
    acceptance_r2score = 0.9
    threshold_accuracy = 0.9

    # If the model is a classifier
    # KNeighborsClassifier does not provide a predict_proba method for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
    if (
        is_classifier_or_partial_classifier(model)
        and get_model_name(model_class) != "KNeighborsClassifier"
    ):
        if is_model_class_in_a_list(model, _get_sklearn_linear_models()):

            # Check outputs from the 'decision_function' method (for linear classifiers)
            y_scores_sklearn = sklearn_model.decision_function(x)
            y_scores_fhe = model.decision_function(x, fhe=fhe)

            assert y_scores_sklearn.shape == y_scores_fhe.shape, (
                "Method 'decision_function' outputs different shapes between scikit-learn and "
                f"Concrete ML in FHE (fhe={fhe})"
            )

            check_r2_score(y_scores_sklearn, y_scores_fhe, acceptance_score=acceptance_r2score)

        # LinearSVC models from scikit-learn do not provide a 'predict_proba' method
        if get_model_name(model_class) != "LinearSVC":

            # Check outputs from the 'predict_proba' method (for all classifiers,
            # except KNeighborsClassifier)
            y_proba_sklearn = sklearn_model.predict_proba(x)
            y_proba_fhe = model.predict_proba(x, fhe=fhe)

            assert y_proba_sklearn.shape == y_proba_fhe.shape, (
                "Method 'decision_function' outputs different shapes between scikit-learn and "
                f"Concrete ML in FHE (fhe={fhe})"
            )
            check_r2_score(y_proba_sklearn, y_proba_fhe, acceptance_score=acceptance_r2score)

    # Check outputs from the 'predict_proba' method (for all models)
    y_pred_sklearn = sklearn_model.predict(x)
    y_pred_fhe = model.predict(x, fhe=fhe)

    assert y_pred_sklearn.shape == y_pred_fhe.shape, (
        "Method 'predict' outputs different shapes between scikit-learn and "
        f"Concrete ML in FHE (fhe={fhe})"
    )

    # If the model is a classifier, check that accuracies are similar
    if is_classifier_or_partial_classifier(model):
        check_accuracy(y_pred_sklearn, y_pred_fhe, threshold=threshold_accuracy)

    # If the model is a regressor, check that R2 scores are similar
    elif is_regressor_or_partial_regressor(model):
        check_r2_score(y_pred_sklearn, y_pred_fhe, acceptance_score=acceptance_r2score)

    else:
        raise AssertionError(f"Model {model_name} is neither a classifier nor a regressor.")


def check_double_fit(model_class, n_bits, x_1, x_2, y_1, y_2):
    """Check double fit."""

    if get_model_name(model_class) == "KNeighborsClassifier":
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4014
        pytest.skip(
            "Given that KNN is not accurate and the test data-set is small"
            "the y_pred1 and y_pred2 can be equal."
        )

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Set the torch seed manually before fitting a neural network
        if is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):

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
        if is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
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
            _get_sklearn_tree_models(classifier=True, select=["RandomForest", "DecisionTree"]),
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
        if is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
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

        # Check if the serialized models are identical
        assert serialized_model_dict == re_serialized_model_dict

        # Check that the predictions made by both model are identical
        y_pred_model = model.predict(x)
        y_pred_loaded_model = loaded_model.predict(x)
        assert numpy.array_equal(y_pred_model, y_pred_loaded_model)

        # Check that the predictions made by both Scikit-Learn model are identical
        y_pred_sklearn_model = model.sklearn_model.predict(x)
        y_pred_loaded_sklearn_model = loaded_model.sklearn_model.predict(x)
        assert numpy.array_equal(y_pred_sklearn_model, y_pred_loaded_sklearn_model)

        # Add a test to check that graphs before and after the serialization are identical
        # FIME: https://github.com/zama-ai/concrete-ml-internal/issues/4175


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

    # Check if the serialized models are identical
    assert serialized_model_dict == re_serialized_model_dict

    # Check that the predictions made by both model are identical
    y_pred_model = model.predict(x)
    y_pred_loaded_model = loaded_model.predict(x)
    assert numpy.array_equal(y_pred_model, y_pred_loaded_model)

    # Check that the predictions made by both Scikit-Learn model are identical
    y_pred_sklearn_model = model.sklearn_model.predict(x)
    y_pred_loaded_sklearn_model = loaded_model.sklearn_model.predict(x)
    assert numpy.array_equal(y_pred_sklearn_model, y_pred_loaded_sklearn_model)

    # Add a test to check that graphs before and after the serialization are identical
    # FIME: https://github.com/zama-ai/concrete-ml-internal/issues/4175


def check_offset(model_class, n_bits, x, y):
    """Check offset."""
    model = instantiate_model_generic(model_class, n_bits=n_bits)

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


def check_inference_methods(model, model_class, x, check_float_array_equal):
    """Check that all inference methods provided are coherent between clear and FHE executions."""

    # skorch provides a predict_proba method for neural network regressors while Scikit-Learn does
    # not. We decided to follow Scikit-Learn's API as we build most of our tools on this library.
    # However, our models are still directly inheriting from skorch's classes, which makes this
    # method accessible by anyone, without having any FHE implementation. As this could create some
    # confusion, a NotImplementedError is raised. This issue could be fixed by making these classes
    # not inherit from skorch.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
    if get_model_name(model) == "NeuralNetRegressor":
        with pytest.raises(
            NotImplementedError,
            match=(
                "The `predict_proba` method is not implemented for neural network regressors. "
                "Please call `predict` instead."
            ),
        ):
            model.predict_proba(x)

    # KNeighborsClassifier does not provide a predict_proba method for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
    elif get_model_name(model) == "KNeighborsClassifier":
        with pytest.raises(
            NotImplementedError,
            match=(
                "The `predict_proba` method is not implemented for KNeighborsClassifier. "
                "Please call `predict` instead."
            ),
        ):
            model.predict_proba(x)

        # KNeighborsClassifier does not provide a kneighbors method
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4080
        with pytest.raises(
            NotImplementedError,
            match=(
                "The `kneighbors` method is not implemented for KNeighborsClassifier. Please call "
                "`get_topk_labels` to retieve the K-Nearest labels for each point, or `predict` "
                "method to retieve the predicted label for each data point."
            ),
        ):
            model.kneighbors(x)

    # Only check 'predict_proba' and not 'predict' as some issues were found with the argmax not
    # being consistent because of precision errors with epsilon magnitude. This argmax should be
    # done in the clear the same way for both anyway. Ultimately, we would want to only compare the
    # circuit's quantized outputs against the ones computed in the clear but built-in models do not
    # currently provide the necessary API for that
    elif is_classifier_or_partial_classifier(model_class):

        if is_model_class_in_a_list(model_class, _get_sklearn_linear_models()):

            # Check outputs from the 'decision_function' method (for all linear classifiers)
            y_scores_clear = model.decision_function(x)
            y_scores_simulated = model.decision_function(x, fhe="simulate")

            assert y_scores_clear.shape == y_scores_simulated.shape, (
                "Method 'decision_function' from Concrete ML outputs different shapes when executed"
                "in the clear and with simulation."
            )
            check_float_array_equal(y_scores_clear, y_scores_simulated)

        else:
            # Check outputs from the 'predict_proba' method (for all non-linear classifiers,
            # except KNeighborsClassifier)
            y_proba_clear = model.predict_proba(x)
            y_proba_simulated = model.predict_proba(x, fhe="simulate")

            assert y_proba_clear.shape == y_proba_simulated.shape, (
                "Method 'predict_proba' from Concrete ML outputs different shapes when executed"
                "in the clear and with simulation."
            )
            check_float_array_equal(y_proba_clear, y_proba_simulated)

    else:
        # Check outputs from the 'predict' method (for all regressors and KNeighborsClassifier)
        y_pred_clear = model.predict(x)
        y_pred_simulated = model.predict(x, fhe="simulate")

        assert y_pred_clear.shape == y_pred_simulated.shape, (
            "Method 'predict' from Concrete ML outputs different shapes when executed in the clear "
            "and with simulation."
        )
        check_float_array_equal(y_pred_clear, y_pred_simulated)


def check_separated_inference(model, fhe_circuit, x, check_float_array_equal):
    """Run inference methods in separated steps and check their correctness."""

    # Generate the keys
    fhe_circuit.keygen()

    # Quantize an input (float)
    q_x = model.quantize_input(x)

    q_y_pred_list = []
    for q_x_i in q_x:
        # Expected input shape for 'encrypt' method is (1, n_features) while q_x_i
        # is of shape (n_features,)
        q_x_i = numpy.expand_dims(q_x_i, 0)

        # Encrypt the input
        q_x_encrypted_i = fhe_circuit.encrypt(q_x_i)

        # Execute the linear product in FHE
        q_y_pred_encrypted_i = fhe_circuit.run(q_x_encrypted_i)

        # Decrypt the result (integer)
        q_y_pred_i = fhe_circuit.decrypt(q_y_pred_encrypted_i)

        q_y_pred_list.append(q_y_pred_i[0])

    q_y_pred = numpy.array(q_y_pred_list)

    # De-quantize the result
    y_pred = model.dequantize_output(q_y_pred)

    if is_model_class_in_a_list(
        model, _get_sklearn_linear_models(classifier=True, regressor=False)
    ):
        y_scores = model.decision_function(x, fhe="simulate")

        # For linear classifiers, the circuit's de-quantized outputs should be the same as the ones
        # from the `decision_function` built-in method
        check_float_array_equal(y_pred, y_scores)

    # Apply post-processing step (in the clear)
    # This includes (non-exhaustive):
    # - sigmoid or softmax function for classifiers
    # - final sum for tree-based models
    # - link function for GLMs
    y_pred = model.post_processing(y_pred)

    # KNeighborsClassifier does not provide a predict_proba method for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
    if (
        is_classifier_or_partial_classifier(model)
        and get_model_name(model) != "KNeighborsClassifier"
    ):
        y_proba = model.predict_proba(x, fhe="simulate")
    else:
        y_proba = model.predict(x, fhe="simulate")

    # The circuit's de-quantized outputs followed by `post_processing` should be the same as the
    # ones from the `predict_proba` built-in method for classifiers, and from the `predict`
    # built-in method for regressors
    check_float_array_equal(y_pred, y_proba)

    # KNeighborsClassifier does not apply a final argmax for computing prediction
    if (
        is_classifier_or_partial_classifier(model)
        and get_model_name(model) != "KNeighborsClassifier"
    ):
        # For linear classifiers, the argmax is done on the scores directly, not the probabilities
        # Also, it is handled differently if shape is (n,) instead of (n, 1)
        if is_model_class_in_a_list(model, _get_sklearn_linear_models()):
            if y_scores.ndim == 1:
                y_pred = (y_scores > 0).astype(int)
            else:
                y_pred = numpy.argmax(y_scores, axis=1)
        else:
            y_pred = numpy.argmax(y_pred, axis=1)

        y_pred_class = model.predict(x, fhe="simulate")

        # For classifiers (other than KNeighborsClassifier), the circuit's de-quantized outputs
        # followed by `post_processing` as well as an argmax should be the same as the ones from
        # the `predict` built-in method
        check_float_array_equal(y_pred, y_pred_class)


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
    # KNeighborsClassifier does not provide a predict_proba method for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
    if (
        is_classifier_or_partial_classifier(model)
        and get_model_name(model_class) != "KNeighborsClassifier"
    ):
        model.predict_proba(x)

    model.compile(x, default_configuration)

    # Make sure `predict` is working when FHE is disabled
    model.predict(x, fhe="simulate")

    # Similarly, we test `predict_proba` for classifiers
    # KNeighborsClassifier does not provide a predict_proba method for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
    if (
        is_classifier_or_partial_classifier(model)
        and get_model_name(model_class) != "KNeighborsClassifier"
    ):
        model.predict_proba(x, fhe="simulate")


def check_pipeline(model_class, x, y):
    """Check pipeline support."""
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
    if is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
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
    if is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
        param_grid = {
            "module__n_layers": [2, 3],
            "module__n_hidden_neurons_multiplier": [1],
            "module__activation_function": (nn.ReLU6,),
        }
    elif model_class in _get_sklearn_tree_models(select="DecisionTree"):
        param_grid = {
            "n_bits": [20],
        }
    elif model_class in _get_sklearn_tree_models():
        param_grid = {
            "n_bits": [20],
            "max_depth": [2],
            "n_estimators": [5, 10],
            "n_jobs": [1],
        }
    elif model_class in _get_sklearn_neighbors_models():
        param_grid = {"n_bits": [2], "n_neighbors": [2]}
    else:
        param_grid = {
            "n_bits": [20],
        }

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        # KNeighborsClassifier does not provide a predict_proba method for now
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
        if get_model_name(model_class) == "KNeighborsClassifier" and scoring in [
            "roc_auc",
            "average_precision",
        ]:
            pytest.skip("Skipping predict_proba for KNN, doesn't work for now")

        _ = GridSearchCV(
            model_class(), param_grid, cv=2, scoring=scoring, error_score="raise", n_jobs=1
        ).fit(x, y)


def get_hyper_param_combinations(model_class):
    """Return the hyper_param_combinations, depending on the model class"""
    hyper_param_combinations: Dict[str, List[Any]]

    if is_model_class_in_a_list(model_class, _get_sklearn_linear_models()):
        hyper_param_combinations = {"fit_intercept": [False, True]}
    elif model_class in _get_sklearn_tree_models(select="DecisionTree"):
        hyper_param_combinations = {}
    elif model_class in _get_sklearn_tree_models(select="RandomForest"):
        hyper_param_combinations = {
            "max_depth": [3, 4, 5, 10],
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 3, 4],
            "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3],
            "max_features": ["sqrt", "log2"],
            "max_leaf_nodes": [None, 5, 10, 20],
        }
    elif model_class in _get_sklearn_tree_models(select="XGB"):
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
    elif model_class in _get_sklearn_neighbors_models():
        # Use small `n_neighbors` values for KNN, because the data-set is too small for now
        hyper_param_combinations = {"n_neighbors": [1, 2]}
    else:

        assert is_model_class_in_a_list(
            model_class, _get_sklearn_neural_net_models()
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

        # Also fit with these hyper parameters to check it works fine
        with warnings.catch_warnings():
            # Sometimes, we miss convergence, which is not a problem for our test
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            # Here, we really need to fit, to take into account hyper parameters
            model.fit(x, y)

        # Check correctness with sklearn
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
    if not is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
        with pytest.raises(AttributeError, match=".* model is not fitted.*"):
            model.predict(x)

    # KNeighborsClassifier does not provide a predict_proba method for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
    if (
        is_classifier_or_partial_classifier(model_class)
        and get_model_name(model) != "KNeighborsClassifier"
    ):

        # Predicting probabilities using an untrained linear or tree-based classifier should not
        # be possible
        if not is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
            with pytest.raises(AttributeError, match=".* model is not fitted.*"):
                model.predict_proba(x)

        # Predicting probabilities in FHE using an untrained QNN classifier should not be possible
        else:
            with pytest.raises(AttributeError, match=".* model is not fitted.*"):
                model.predict_proba(x, fhe="execute")

        # Computing the decision function using an untrained classifier should not be possible.
        # Note that the `decision_function` method is only available for linear models
        if is_model_class_in_a_list(model_class, _get_sklearn_linear_models()):
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
        model_class, _get_sklearn_neural_net_models()
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


def check_load_fitted_sklearn_linear_models(model_class, n_bits, x, y, check_float_array_equal):
    """Check that linear models and QNNs support loading from pre-trained scikit-learn models."""

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    # Fit the model and retrieve both the Concrete ML and the scikit-learn models
    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        concrete_model, sklearn_model = model.fit_benchmark(x, y)

    # This step is needed in order to handle partial classes
    model_class = get_model_class(model_class)

    # Load a Concrete ML model from the fitted scikit-learn one
    loaded_concrete_model = model_class.from_sklearn_model(sklearn_model, X=x, n_bits=n_bits)

    # Compile both the initial Concrete ML model and the loaded one
    concrete_model.compile(x)
    loaded_concrete_model.compile(x)

    # Compute and compare the predictions from both models
    y_pred_simulate = concrete_model.predict(x, fhe="simulate")
    y_pred_simulate_loaded = loaded_concrete_model.predict(x, fhe="simulate")

    check_float_array_equal(
        y_pred_simulate,
        y_pred_simulate_loaded,
        error_information="Simulated predictions from the initial model do not match the ones made "
        "from the loaded one.",
    )


def check_rounding_consistency(
    model,
    x,
    y,
    predict_method,
    metric,
    is_weekly_option,
):
    """Test that Concrete ML without and with rounding are 'equivalent'."""

    # Run the test with more samples during weekly CIs
    if is_weekly_option:
        fhe_test = get_random_samples(x, n_sample=5)

    # Check that rounding is enabled
    assert os.environ.get("TREES_USE_ROUNDING") == "1", "'TREES_USE_ROUNDING' is not enabled"

    # Fit and compile with rounding enabled
    fit_and_compile(model, x, y)

    rounded_predict_quantized = predict_method(x, fhe="disable")
    rounded_predict_simulate = predict_method(x, fhe="simulate")

    # Compute the FHE predictions only during weekly CIs
    if is_weekly_option:
        rounded_predict_fhe = predict_method(fhe_test, fhe="execute")

    with pytest.MonkeyPatch.context() as mp_context:

        # Disable rounding
        mp_context.setenv("TREES_USE_ROUNDING", "0")

        # Check that rounding is disabled
        assert os.environ.get("TREES_USE_ROUNDING") == "0", "'TREES_USE_ROUNDING' is not disabled"

        with pytest.warns(
            DeprecationWarning,
            match=(
                "Using Concrete tree-based models without the `rounding feature` is " "deprecated.*"
            ),
        ):

            # Fit and compile without rounding
            fit_and_compile(model, x, y)

        not_rounded_predict_quantized = predict_method(x, fhe="disable")
        not_rounded_predict_simulate = predict_method(x, fhe="simulate")

        metric(rounded_predict_quantized, not_rounded_predict_quantized)
        metric(rounded_predict_simulate, not_rounded_predict_simulate)

        # Compute the FHE predictions only during weekly CIs
        if is_weekly_option:
            not_rounded_predict_fhe = predict_method(fhe_test, fhe="execute")
            metric(rounded_predict_fhe, not_rounded_predict_fhe)

        # Check that the maximum bit-width of the circuit with rounding is at most:
        # maximum bit-width (of the circuit without rounding) + 2
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4178


def check_sum_for_tree_based_models(
    model,
    x,
    y,
    predict_method,
    is_weekly_option,
):
    """Test that Concrete ML without and with FHE sum are 'equivalent'."""

    fhe_samples = 5
    fhe_test = get_random_samples(x, n_sample=fhe_samples)

    # pylint: disable=protected-access
    assert not model._fhe_ensembling, "`_fhe_ensembling` is disabled by default."
    fit_and_compile(model, x, y)

    non_fhe_sum_predict_quantized = predict_method(x, fhe="disable")
    non_fhe_sum_predict_simulate = predict_method(x, fhe="simulate")

    if is_weekly_option:
        non_fhe_sum_predict_fhe = predict_method(fhe_test, fhe="execute")

    # Sanity check
    array_allclose_and_same_shape(non_fhe_sum_predict_quantized, non_fhe_sum_predict_simulate)

    # pylint: disable=protected-access
    model._fhe_ensembling = True

    fit_and_compile(model, x, y)

    fhe_sum_predict_quantized = predict_method(x, fhe="disable")
    fhe_sum_predict_simulate = predict_method(x, fhe="simulate")

    if is_weekly_option:
        fhe_sum_predict_fhe = predict_method(fhe_test, fhe="execute")

    # Sanity check
    array_allclose_and_same_shape(fhe_sum_predict_quantized, fhe_sum_predict_simulate)

    # Check that we have the exact same predictions
    array_allclose_and_same_shape(fhe_sum_predict_quantized, non_fhe_sum_predict_quantized)
    array_allclose_and_same_shape(fhe_sum_predict_simulate, non_fhe_sum_predict_simulate)
    if is_weekly_option:
        array_allclose_and_same_shape(fhe_sum_predict_fhe, non_fhe_sum_predict_fhe)


# Neural network models are skipped for this test
# The `fit_benchmark` function of QNNs returns a QAT model and a FP32 model that is similar
# in structure but trained from scratch. Furthermore, the `n_bits` setting
# of the QNN instantiation in `instantiate_model_generic` takes `n_bits` as
# a target accumulator and sets 3-b w&a for these tests. Thus it's
# impossible to reach R-2 of 0.99 when comparing the two NN models returned by `fit_benchmark`
@pytest.mark.parametrize(
    "model_class, parameters",
    get_sklearn_linear_models_and_datasets()
    + get_sklearn_tree_models_and_datasets()
    + get_sklearn_neighbors_models_and_datasets(),
)
def test_correctness_with_sklearn(
    model_class,
    parameters,
    load_data,
    check_r2_score,
    check_accuracy,
    is_weekly_option,
    verbose=True,
):
    """Test that Concrete ML and scikit-learn models are 'equivalent'."""

    n_bits = N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

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


# Neural network hyper-parameters are not tested
@pytest.mark.parametrize(
    "model_class, parameters",
    get_sklearn_linear_models_and_datasets()
    + get_sklearn_tree_models_and_datasets()
    + get_sklearn_neighbors_models_and_datasets(),
)
def test_hyper_parameters(
    model_class,
    parameters,
    load_data,
    check_r2_score,
    check_accuracy,
    is_weekly_option,
    verbose=True,
):
    """Testing hyper parameters."""

    n_bits = N_BITS_THRESHOLD_FOR_SKLEARN_CORRECTNESS_TESTS

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_hyper_parameters")

    check_hyper_parameters(
        model_class,
        n_bits,
        x,
        y,
        check_r2_score,
        check_accuracy,
    )


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
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


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
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
    n_bits = get_n_bits_non_correctness(model_class)

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Compile the model to make sure we consider all possible attributes during the serialization
    model.compile(x, default_configuration)

    if verbose:
        print("Run check_serialization")

    check_serialization(model, x, use_dump_method)


@pytest.mark.parametrize("model_class, parameters", UNIQUE_MODELS_AND_DATASETS)
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


# Offsets are not supported by XGBoost models
@pytest.mark.parametrize(
    "model_class, parameters",
    get_sklearn_all_models_and_datasets(ignore="XGB", unique_models=True),
)
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


@pytest.mark.parametrize("model_class, parameters", UNIQUE_MODELS_AND_DATASETS)
@pytest.mark.parametrize("input_type", ["numpy", "torch", "pandas", "list"])
def test_input_support(
    model_class,
    parameters,
    load_data,
    input_type,
    default_configuration,
    is_weekly_option,
    verbose=True,
):
    """Test all models with Pandas, List or Torch inputs."""
    n_bits = get_n_bits_non_correctness(model_class)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run input_support")

    check_input_support(model_class, n_bits, default_configuration, x, y, input_type)


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
def test_inference_methods(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    check_float_array_equal,
    default_configuration,
    verbose=True,
):
    """Test inference methods."""
    n_bits = get_n_bits_non_correctness(model_class)

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    model.compile(x, default_configuration)

    if verbose:
        print("Run check_inference_methods")

    check_inference_methods(model, model_class, x, check_float_array_equal)


# Pipeline test sometimes fails with RandomForest models. This bug may come from Hummingbird
# and needs further investigations
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2779
@pytest.mark.parametrize(
    "model_class, parameters", get_sklearn_all_models_and_datasets(ignore="RandomForest")
)
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


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
@pytest.mark.parametrize(
    "simulate",
    [
        pytest.param(False, id="fhe"),
        pytest.param(True, id="simulate"),
    ],
)
# N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS bits is currently the
# limit to find crypto parameters for linear models
# make sure we only compile below that bit-width.
# Additionally, prevent computations in FHE with too many bits
@pytest.mark.parametrize(
    "n_bits",
    [
        n_bits
        for n_bits in N_BITS_WEEKLY_ONLY_BUILDS + N_BITS_REGULAR_BUILDS
        if n_bits
        < min(N_BITS_LINEAR_MODEL_CRYPTO_PARAMETERS, N_BITS_THRESHOLD_TO_FORCE_EXECUTION_NOT_IN_FHE)
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
    is_weekly_option,
    verbose=True,
):
    """Test prediction correctness between clear quantized and FHE simulation or execution."""

    # KNN can only be compiled with small quantization bit numbers for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3979
    if n_bits > 5 and get_model_name(model_class) == "KNeighborsClassifier":
        pytest.skip("KNeighborsClassifier models can only run with 5 bits at most.")

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)
    # Run the test with more samples during weekly CIs or when using FHE simulation
    if is_weekly_option or simulate:
        fhe_samples = 5
    else:
        fhe_samples = 1

    if verbose:
        print("Compile the model")

    model.compile(x, default_configuration)

    if verbose:
        print(f"Check prediction correctness for {fhe_samples} samples.")

    # Check prediction correctness between quantized clear and FHE simulation or execution
    fhe_test = get_random_samples(x, fhe_samples)
    check_is_good_execution_for_cml_vs_circuit(fhe_test, model=model, simulate=simulate)


@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
# pylint: disable=too-many-branches
def test_separated_inference(
    model_class,
    parameters,
    load_data,
    default_configuration,
    is_weekly_option,
    check_float_array_equal,
    verbose=True,
):
    """Test prediction correctness between clear quantized and FHE simulation or execution."""

    n_bits = min(N_BITS_REGULAR_BUILDS)

    # KNN can only be compiled with small quantization bit numbers for now
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3979
    if n_bits > 5 and get_model_name(model_class) == "KNeighborsClassifier":
        pytest.skip("KNeighborsClassifier models can only run with 5 bits at most.")

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Run the test with more samples during weekly CIs or when using FHE simulation
    if is_weekly_option:
        fhe_samples = 5
    else:
        fhe_samples = 1

    if verbose:
        print("Compile the model")

    fhe_circuit = model.compile(x, default_configuration)

    if verbose:
        print("Run check_separated_inference")

    # Check that separated inference steps (encrypt, run, decrypt, post_processing, ...) are
    # equivalent to built-in methods (predict, predict_proba, ...)
    fhe_test = get_random_samples(x, fhe_samples)
    check_separated_inference(model, fhe_circuit, fhe_test, check_float_array_equal)


@pytest.mark.parametrize("model_class, parameters", UNIQUE_MODELS_AND_DATASETS)
def test_fitted_compiled_error_raises(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test Fit and Compile error raises."""
    n_bits = get_n_bits_non_correctness(model_class)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_fitted_compiled_error_raises")

    check_fitted_compiled_error_raises(model_class, n_bits, x, y)


# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4169
@pytest.mark.flaky
@pytest.mark.parametrize("model_class, parameters", MODELS_AND_DATASETS)
@pytest.mark.parametrize(
    "error_param, expected_diff",
    [({"p_error": 1 - 2**-40}, True), ({"p_error": 2**-40}, False)],
    ids=["p_error_high", "p_error_low"],
)
def test_p_error_simulation(
    model_class,
    parameters,
    error_param,
    expected_diff,
    load_data,
    is_weekly_option,
):
    """Test p_error simulation.

    The test checks that models compiled with a large p_error value predicts very different results
    with simulation or in FHE compared to the expected clear quantized ones.
    """

    n_bits = get_n_bits_non_correctness(model_class)

    # Get data-set, initialize and fit the model
    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Compile with the specified p_error.
    model.compile(x, **error_param)

    def check_for_divergent_predictions(
        x, model, fhe, max_iterations=N_ALLOWED_FHE_RUN, tolerance=1e-5
    ):
        """Detect divergence between simulated/FHE execution and clear run."""

        # KNeighborsClassifier does not provide a predict_proba method for now
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962
        predict_function = (
            model.predict_proba
            if is_classifier_or_partial_classifier(model)
            and get_model_name(model) != "KNeighborsClassifier"
            else model.predict
        )

        y_expected = predict_function(x, fhe="disable")
        for i in range(max_iterations):
            y_pred = predict_function(x[i : i + 1], fhe=fhe).ravel()
            if not numpy.allclose(y_pred, y_expected[i : i + 1].ravel(), atol=tolerance):
                return True
        return False

    simulation_diff_found = check_for_divergent_predictions(x, model, fhe="simulate")
    fhe_diff_found = check_for_divergent_predictions(x, model, fhe="execute")

    # Check if model is linear
    is_linear_model = is_model_class_in_a_list(model_class, _get_sklearn_linear_models())

    # Skip the following if model is linear
    # Simulation and FHE differs with very high p_error on leveled circuit
    # FIXME https://github.com/zama-ai/concrete-ml-internal/issues/4343
    if is_linear_model:
        pytest.skip("Skipping test for linear models")

    # Check for differences in predictions based on expected_diff
    if expected_diff:
        assert_msg = (
            "With high p_error, predictions should differ in both FHE and simulation."
            f" Found differences: FHE={fhe_diff_found}, Simulation={simulation_diff_found}"
        )
        assert fhe_diff_found and simulation_diff_found, assert_msg
    else:
        assert_msg = (
            "With low p_error, predictions should not differ in FHE or simulation."
            f" Found differences: FHE={fhe_diff_found}, Simulation={simulation_diff_found}"
        )
        assert not (fhe_diff_found or simulation_diff_found), assert_msg


# This test is only relevant for classifier models
@pytest.mark.parametrize(
    "model_class, parameters", get_sklearn_all_models_and_datasets(regressor=False, classifier=True)
)
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


@pytest.mark.parametrize("model_class, parameters", UNIQUE_MODELS_AND_DATASETS)
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


@pytest.mark.parametrize(
    "model_class, parameters", get_sklearn_tree_models_and_datasets(select="DecisionTree")
)
def test_exposition_structural_methods_decision_trees(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Test the exposition of specific structural methods found in decision tree models."""
    n_bits = min(N_BITS_REGULAR_BUILDS)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    if verbose:
        print("Run check_exposition_structural_methods_decision_trees")

    check_exposition_structural_methods_decision_trees(model, x, y)


# Importing fitted models only works with linear models
@pytest.mark.parametrize("model_class, parameters", get_sklearn_linear_models_and_datasets())
def test_load_fitted_sklearn_linear_models(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    check_float_array_equal,
    verbose=True,
):
    """Test that linear models support loading from fitted scikit-learn models."""

    n_bits = min(N_BITS_REGULAR_BUILDS)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Run check_load_pre_trained_sklearn_models")

    check_load_fitted_sklearn_linear_models(model_class, n_bits, x, y, check_float_array_equal)


# Only circuits from linear models do not have any TLUs
@pytest.mark.parametrize("model_class, parameters", get_sklearn_linear_models_and_datasets())
def test_linear_models_have_no_tlu(
    model_class,
    parameters,
    load_data,
    is_weekly_option,
    check_circuit_has_no_tlu,
    default_configuration,
    verbose=True,
):
    """Test that circuits from linear models have no TLUs."""

    n_bits = min(N_BITS_REGULAR_BUILDS)

    model, x = preamble(model_class, parameters, n_bits, load_data, is_weekly_option)

    if verbose:
        print("Compile the model")

    fhe_circuit = model.compile(x, default_configuration)

    if verbose:
        print("Run check_circuit_has_no_tlu")

    # Check that no TLUs are found within the MLIR
    check_circuit_has_no_tlu(fhe_circuit)


# This test does not check rounding at level 2
# Additional tests for this purpose should be added in future updates
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4179
@pytest.mark.parametrize("model_class, parameters", get_sklearn_tree_models_and_datasets())
@pytest.mark.parametrize("n_bits", [2, 5, 10])
def test_rounding_consistency_for_regular_models(
    model_class,
    parameters,
    n_bits,
    load_data,
    check_r2_score,
    is_weekly_option,
    verbose=True,
):
    """Test that Concrete ML without and with rounding are 'equivalent'."""

    if verbose:
        print("Run check_rounding_consistency")

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    # Check `predict_proba` for classifiers
    if is_classifier_or_partial_classifier(model):
        predict_method = model.predict_proba
        metric = check_r2_score
    else:
        # Check `predict` for regressors
        predict_method = model.predict
        metric = check_r2_score

    check_rounding_consistency(
        model,
        x,
        y,
        predict_method,
        metric,
        is_weekly_option,
    )


@pytest.mark.parametrize("model_class, parameters", get_sklearn_tree_models_and_datasets())
@pytest.mark.parametrize("n_bits", [2, 5, 10])
@pytest.mark.parametrize("execute_in_fhe", [True, False])
def test_sum_for_tree_based_models(
    model_class,
    parameters,
    n_bits,
    load_data,
    is_weekly_option,
    execute_in_fhe,
    verbose=True,
):
    """Test that the tree ensembles' output are the same with and without the sum in FHE."""

    if execute_in_fhe and not is_weekly_option:
        pytest.skip("Skipping FHE tests in non-weekly builds")

    if verbose:
        print("Run check_fhe_sum_for_tree_based_models")

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    predict_method = (
        model.predict_proba if is_classifier_or_partial_classifier(model) else model.predict
    )
    check_sum_for_tree_based_models(
        model=model,
        x=x,
        y=y,
        predict_method=predict_method,
        is_weekly_option=is_weekly_option,
    )


# This test should be extended to all built-in models.
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4234
@pytest.mark.parametrize(
    "n_bits, error_message",
    [
        (0, "n_bits must be a strictly positive integer"),
        (-1, "n_bits must be a strictly positive integer"),
        ({"op_leaves": 2}, "The key 'op_inputs' is mandatory"),
        (
            {"op_inputs": 4, "op_leaves": 2, "op_weights": 2},
            "Invalid keys in 'n_bits' dictionary. Only 'op_inputs' \\(mandatory\\) and 'op_leaves' "
            "\\(optional\\) are allowed",
        ),
        (
            {"op_inputs": -2, "op_leaves": -5},
            "All values in 'n_bits' dictionary must be strictly positive integers",
        ),
        ({"op_inputs": 2, "op_leaves": 5}, "'op_leaves' must be less than or equal to 'op_inputs'"),
        (0.5, "n_bits must be either an integer or a dictionary"),
    ],
)
@pytest.mark.parametrize("model_class", _get_sklearn_tree_models())
def test_invalid_n_bits_setting(model_class, n_bits, error_message):
    """Check if the model instantiation raises an exception with invalid `n_bits` settings."""

    with pytest.raises(ValueError, match=f"{error_message}. Got '{type(n_bits)}' and '{n_bits}'.*"):
        instantiate_model_generic(model_class, n_bits=n_bits)


# This test should be extended to all built-in models.
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4234
@pytest.mark.parametrize("n_bits", [5, {"op_inputs": 5}, {"op_inputs": 2, "op_leaves": 1}])
@pytest.mark.parametrize("model_class, parameters", get_sklearn_tree_models_and_datasets())
def test_valid_n_bits_setting(
    model_class,
    n_bits,
    parameters,
    load_data,
    is_weekly_option,
    verbose=True,
):
    """Check valid `n_bits` settings."""

    if verbose:
        print("Run test_valid_n_bits_setting")

    x, y = get_dataset(model_class, parameters, n_bits, load_data, is_weekly_option)

    model = instantiate_model_generic(model_class, n_bits=n_bits)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)


# A type error will be raised for NeuralNetworks, which is tested in test_failure_bad_data_types
@pytest.mark.parametrize(
    "model_class",
    _get_sklearn_linear_models() + _get_sklearn_tree_models() + _get_sklearn_neighbors_models(),
)
@pytest.mark.parametrize(
    "bad_value, expected_error",
    [
        (numpy.nan, "Input X contains NaN."),
        (None, "Input X contains NaN."),
        ("this", "could not convert string to float: 'this'"),
    ],
)
def test_error_raise_unsupported_pandas_values(model_class, bad_value, expected_error):
    """Test that using Pandas data-frame with unsupported values as input raises correct errors."""

    dic = {
        "Col One": [1, 2, bad_value, 3],
        "Col Two": [4, 5, 6, bad_value],
        "Col Three": [bad_value, 7, 8, 9],
    }

    # Creating a dataframe using dictionary
    x_train = pandas.DataFrame(dic)
    y_train = x_train["Col Three"]

    model = model_class(n_bits=2)

    # The error message changed in one of our dependencies
    assert sys.version_info.major == 3
    if sys.version_info.minor <= 7:
        if expected_error == "Input X contains NaN.":
            expected_error = "Input contains NaN*"

    with pytest.raises(ValueError, match=expected_error):
        model.fit(x_train, y_train)
