"""Test binary search class."""

import os
import warnings
from pathlib import Path

import numpy
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score, top_k_accuracy_score
from tensorflow import keras

from concrete.ml.common.check_inputs import check_array_and_assert
from concrete.ml.common.utils import get_model_name, is_regressor_or_partial_regressor
from concrete.ml.pytest.torch_models import QuantCustomModel, TorchCustomModel
from concrete.ml.pytest.utils import (
    data_calibration_processing,
    get_sklearn_linear_models_and_datasets,
    get_sklearn_neighbors_models_and_datasets,
    get_sklearn_neural_net_models_and_datasets,
    get_sklearn_tree_models_and_datasets,
    instantiate_model_generic,
    load_torch_model,
)
from concrete.ml.search_parameters import BinarySearch

# For built-in models (trees and QNNs) we use the fixture `load_data`
# For custom models, we define the following variables:
DATASETS_ARGS = {
    "CustomClassificationDataset": {
        "n_samples": 1000,
        "n_classes": 3,
        "n_features": 6,
        "n_informative": 4,
        "n_redundant": 2,
        "random_state": 42,
    },
}

TEST_DATA_DIR = Path(__file__).parents[1] / "data" / "parameter_search"

MODELS_ARGS = {
    "CustomModel": {
        "qat": {
            "model_class": QuantCustomModel,
            "path": TEST_DATA_DIR / "custom_data_quant_state_dict.pt",
            "params": {"n_bits": 4, "input_shape": 6, "hidden_shape": 100, "output_shape": 3},
        },
        "fp32": {
            "model_class": TorchCustomModel,
            "path": torch.load(
                (TEST_DATA_DIR / "custom_data_fp32_state_dict.pt"), map_location="cpu"
            ),
            "params": {"input_shape": 6, "hidden_shape": 100, "output_shape": 3},
        },
        "dataset": DATASETS_ARGS["CustomClassificationDataset"],
    },
}


def binary_classification_metric(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
    """Binary classification metric.
    Args:
        y_true (numpy.ndarray): Ground truth
        y_pred (numpy.ndarray): Model predictions.

    Returns:
        float: Accuracy.
    """
    y_true = check_array_and_assert(y_true, ensure_2d=False)
    y_pred = check_array_and_assert(y_pred, ensure_2d=False)
    # Tree models return 1D array unlike NNs
    y_pred = y_pred.argmax(1) if y_pred.ndim > 1 else y_pred
    return (y_pred == y_true).mean()


@pytest.mark.parametrize("attr, value", [("undefined_attr", 1)])
@pytest.mark.parametrize(
    "model_name, quant_type, metric", [("CustomModel", "qat", binary_classification_metric)]
)
def test_update_invalid_attr_method(attr, value, model_name, quant_type, metric, load_data):
    """Check if  `_update_attr` method raises an exception when an unexpected attribute is given."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    x, y = load_data(model, **MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(
        estimator=model,
        metric=metric,
        predict="predict",
    )

    with pytest.raises(AttributeError, match=".* does not belong to this class"):
        search.run(x=x_calib, ground_truth=y, strategy=all, **{attr: value})


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("attr, value", [("max_iter", 1)])
@pytest.mark.parametrize(
    "model_name, quant_type, metric",
    [("CustomModel", "qat", binary_classification_metric)],
)
def test_update_valid_attr_method(attr, value, model_name, quant_type, metric, load_data):
    """Check that `_update_attr` can successfully update given valid attributes."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    x, y = load_data(model, **MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(
        estimator=model,
        metric=metric,
        predict="predict",
        n_simulation=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.run(x=x_calib, ground_truth=y, strategy=all, **{attr: value})

    assert getattr(search, attr) == value


# Linear models are not concerned by this test because they do not have PBS
@pytest.mark.parametrize(
    "model_class, parameters",
    get_sklearn_tree_models_and_datasets(unique_models=True)
    + get_sklearn_neural_net_models_and_datasets(unique_models=True)
    + get_sklearn_neighbors_models_and_datasets(unique_models=True),
)
def test_non_convergence_for_built_in_models(model_class, parameters, load_data, is_weekly_option):
    """Check that binary search raises a user warning when convergence is not achieved.

    The user warning is raised if (1 / n_simulation) ∑ metric_difference_i > max_metric_loss.
    Since p_error represents a probability, we cannot guarantee that the convergence will not be
    reached after only one iteration. Therefore, max_metric_loss is set to a negative number.
    """

    if not is_weekly_option:
        pytest.skip("Tests too long")

    x, y = load_data(model_class, **parameters)
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    model = instantiate_model_generic(model_class, n_bits=4)

    metric = r2_score if is_regressor_or_partial_regressor(model) else binary_classification_metric

    search = BinarySearch(
        estimator=model,
        predict="predict",
        metric=metric,
        max_iter=1,
        n_simulation=5,
        max_metric_loss=-10,
        is_qat=False,
    )

    warnings.simplefilter("always")
    with pytest.warns(UserWarning, match="ConvergenceWarning: .*"):
        search.run(x=x_calib, ground_truth=y, strategy=all)


@pytest.mark.parametrize("model_name, quant_type", [("CustomModel", "qat")])
def test_non_convergence_for_custom_models(model_name, quant_type):
    """Check that binary search raises a user warning when convergence is not achieved.

    The user warning is raised if (1 / n_simulation) ∑ metric_difference_i > max_metric_loss.
    Since p_error represents a probability, we cannot guarantee that the convergence will not be
    reached after only one iteration. Therefore, max_metric_loss is set to a negative number.
    """

    # Load pre-trained model
    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    # Load data-set
    x, y = make_classification(**MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(
        estimator=model,
        predict="predict",
        metric=top_k_accuracy_score,
        max_metric_loss=-10,
        max_iter=1,
        n_simulation=1,
        is_qat=quant_type == "qat",
        k=1,
        labels=numpy.arange(MODELS_ARGS[model_name]["dataset"]["n_classes"]),
    )

    warnings.simplefilter("always")
    with pytest.warns(UserWarning, match="ConvergenceWarning: .*"):
        search.run(x=x_calib, ground_truth=y, strategy=all)


@pytest.mark.parametrize(
    "strategy",
    [
        all,
        any,
        lambda all_matches: numpy.mean(all_matches) >= 0.5,
        lambda all_matches: numpy.median(all_matches) == 1,
    ],
)
@pytest.mark.parametrize("model_name, quant_type", [("CustomModel", "qat")])
def test_valid_strategy(strategy, model_name, quant_type):
    """Check some valid strategies."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    search = BinarySearch(estimator=model, predict="predict", metric=top_k_accuracy_score)
    search.eval_match(strategy, all_matches=numpy.random.choice([True, False], size=5))


@pytest.mark.parametrize("strategy", [lambda x, y: x + y, lambda x: x, "funct"])
@pytest.mark.parametrize("model_name, quant_type", [("CustomModel", "qat")])
def test_invalid_strategy(strategy, model_name, quant_type):
    """Check if `eval_match` method raises an exception when a non valid function is given."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    search = BinarySearch(estimator=model, predict="predict", metric=top_k_accuracy_score)

    with pytest.raises(TypeError, match=".* is not valid."):
        search.eval_match(strategy, all_matches=numpy.random.choice([True, False], size=5))


@pytest.mark.parametrize("threshold,", [0.02])
@pytest.mark.parametrize(
    "model_name, quant_type", [("CustomModel", "qat"), ("CustomModel", "fp32")]
)
def test_binary_search_for_custom_models(model_name, quant_type, threshold):
    """Check if the returned `p_error` is valid for custom NNs."""

    # Load pre-trained model
    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    # Load data-set
    x, y = make_classification(**MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=10)

    search = BinarySearch(
        estimator=model,
        predict="predict",
        metric=top_k_accuracy_score,
        max_metric_loss=threshold,
        n_simulation=5,
        max_iter=1,
        is_qat=quant_type == "qat",
        k=1,
        labels=numpy.arange(MODELS_ARGS[model_name]["dataset"]["n_classes"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        largest_perror = search.run(x=x_calib, ground_truth=y, strategy=all)

    assert 1.0 > largest_perror > 0.0
    assert (
        numpy.mean(search.history[-1]["metric_difference"]) <= threshold
        or len(search.history) == search.max_iter
    )


# Linear models are not concerned by this test because they do not have PBS
@pytest.mark.parametrize("threshold", [0.02])
@pytest.mark.parametrize(
    "model_class, parameters",
    get_sklearn_tree_models_and_datasets(unique_models=True)
    + get_sklearn_neural_net_models_and_datasets(unique_models=True)
    + get_sklearn_neighbors_models_and_datasets(unique_models=True),
)
@pytest.mark.parametrize("predict", ["predict", "predict_proba"])
def test_binary_search_for_built_in_models(model_class, parameters, threshold, predict, load_data):
    """Check if the returned `p_error` is valid for built-in models."""

    x, y = load_data(model_class, **parameters)
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=80)

    model = instantiate_model_generic(model_class, n_bits=4)

    # NeuralNetRegressor models support a `predict_proba` method since it directly inherits from
    # Skorch but since Scikit-Learn does not, we don't as well. This issue could be fixed by making
    # neural networks not inherit from Skorch.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
    # Skipping predict_proba for KNN, doesn't work for now.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3962

    if predict == "predict_proba" and get_model_name(model_class) in [
        "NeuralNetRegressor",
        "KNeighborsClassifier",
    ]:
        return

    metric = r2_score if is_regressor_or_partial_regressor(model) else binary_classification_metric

    if hasattr(type(model), predict):

        search = BinarySearch(
            estimator=model,
            predict=predict,
            metric=metric,
            n_simulation=2,
            max_metric_loss=threshold,
            is_qat=False,
            max_iter=2,
        )
    else:
        # The model does not have `predict`
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        largest_perror = search.run(x=x_calib, ground_truth=y, strategy=all)

    assert 1.0 > largest_perror > 0.0
    assert (
        numpy.mean(search.history[-1]["metric_difference"]) <= threshold
        or len(search.history) == search.max_iter
    )


@pytest.mark.parametrize("is_qat", [False, True])
def test_invalid_estimator_for_custom_models(is_qat, load_data):
    """Check that binary search raises an exception for unsupported models."""

    model_name = "CustomModel"
    model = keras.Sequential(
        [
            keras.layers.InputLayer(
                input_shape=(MODELS_ARGS[model_name]["dataset"]["n_features"],)
            ),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=16, activation="relu"),
            keras.layers.Dense(
                units=MODELS_ARGS[model_name]["dataset"]["n_classes"], activation="softmax"
            ),
        ]
    )

    model_for_data = load_torch_model(
        MODELS_ARGS[model_name]["qat"]["model_class"],
        MODELS_ARGS[model_name]["qat"]["path"],
        MODELS_ARGS[model_name]["qat"]["params"],
    )

    x, y = load_data(model_for_data, **MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(
        estimator=model,
        is_qat=is_qat,
        predict="predict",
        metric=top_k_accuracy_score,
    )

    with pytest.raises(ValueError, match=".* is not supported. .*"):
        search.run(x=x_calib, ground_truth=y, strategy=all, max_iter=1, n_simulation=1)


# This test only concerns linear models since they do not contain any PBS
@pytest.mark.parametrize(
    "model_class, parameters",
    get_sklearn_linear_models_and_datasets(unique_models=True),
)
def test_invalid_estimator_for_built_in_models(model_class, parameters, load_data):
    """Check that binary search raises an exception for unsupported models."""

    x, y = load_data(model_class, **parameters)
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    model = instantiate_model_generic(model_class, n_bits=4)

    metric = r2_score if is_regressor_or_partial_regressor(model) else binary_classification_metric

    search = BinarySearch(
        estimator=model,
        predict="predict",
        metric=metric,
        is_qat=False,
    )

    with pytest.raises(ValueError, match=".* is not supported. .*"):
        search.run(x=x_calib, ground_truth=y)


@pytest.mark.parametrize(
    "model_name, quant_type, metric", [("CustomModel", "qat", binary_classification_metric)]
)
def test_failure_save_option(model_name, quant_type, metric):
    """Check that `_update_attr` can successfully update given valid attributes."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    with pytest.raises(AssertionError, match="To save logs, file name and path must be provided"):
        _ = BinarySearch(
            estimator=model,
            predict="predict",
            metric=metric,
            save=True,
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "directory, log_file",
    [
        ("/tmp/binary_search_perror", "log_file_1.txt"),
        (Path("/tmp/binary_search_perror"), "log_file_2.txt"),
        ("/tmp/binary_search_perror", Path("log_file_3.txt")),
        (Path("/tmp/binary_search_perror"), Path("log_file_4.txt")),
    ],
)
@pytest.mark.parametrize(
    "model_name, quant_type, metric", [("CustomModel", "qat", binary_classification_metric)]
)
def test_success_save_option(model_name, quant_type, metric, directory, log_file, load_data):
    """Check that `_update_attr` can successfully update given valid attributes."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    x, y = load_data(model, **MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(
        estimator=model,
        metric=metric,
        predict="predict",
        save=True,
        directory=directory,
        log_file=log_file,
        max_iter=1,
        n_simulation=1,
        verbose=True,
    )

    path = Path(os.path.join(directory, log_file))

    # When instantiating the class, if the file exists, it is deleted, to avoid overwriting it
    assert not path.exists()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.run(x=x_calib, ground_truth=y)

    # Check that the file has been properly created
    assert path.exists()

    # Check that the file contains at least 2 rows (one for the header and another one for the data)
    with path.open(mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 2
