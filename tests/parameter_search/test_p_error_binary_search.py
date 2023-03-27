"""Test binary search class."""

import warnings
from pathlib import Path

import numpy
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import r2_score, top_k_accuracy_score
from tensorflow import keras
from torchvision import datasets, transforms

from concrete.ml.common.check_inputs import check_array_and_assert
from concrete.ml.common.utils import is_model_class_in_a_list, is_regressor_or_partial_regressor
from concrete.ml.pytest.torch_models import QNNFashionMNIST, QuantCustomModel, TorchCustomModel
from concrete.ml.pytest.utils import (
    data_calibration_processing,
    get_torchvision_dataset,
    instantiate_model_generic,
    load_torch_model,
    sklearn_models_and_datasets,
)
from concrete.ml.search_parameters import BinarySearch
from concrete.ml.sklearn import get_sklearn_linear_models

DATASETS_ARGS = {
    "FashionMNIST": {
        "dataset": datasets.FashionMNIST,
        "train_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.2859, 0.3530)]
        ),
        "test_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.2859, 0.3530)]
        ),
        "dataset_size": 60000,
        "n_classes": 10,
    },
    "CustomClassificationDataset": {
        "n_samples": 1000,
        "n_classes": 3,
        "n_features": 6,
        "n_informative": 4,
        "n_redundant": 2,
        "random_state": 42,
    },
}


MODELS_ARGS = {
    "FashionMNIST": {
        "qat": {
            "model_class": QNNFashionMNIST,
            "path": Path(__file__).parent / "FashionMNIST_quant_state_dict.pt",
            "params": {"n_bits": 4},
        },
        "dataset": DATASETS_ARGS["FashionMNIST"],
    },
    "CustomModel": {
        "qat": {
            "model_class": QuantCustomModel,
            "path": Path(__file__).parent / "custom_data_quant_state_dict.pt",
            "params": {"n_bits": 4, "input_shape": 6, "hidden_shape": 100, "output_shape": 3},
        },
        "fp32": {
            "model_class": TorchCustomModel,
            "path": torch.load(
                (Path(__file__).parent / "custom_data_fp32_state_dict.pt"), map_location="cpu"
            ),
            "params": {"input_shape": 6, "hidden_shape": 100, "output_shape": 3},
        },
        "dataset": DATASETS_ARGS["CustomClassificationDataset"],
    },
}


def binary_classification_metric(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
    """Binary classifcation metric.
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
    )

    with pytest.raises(AttributeError, match=".* does not belong to this class"):
        search.run(x=x_calib, ground_truth=y, strategy=all, **{attr: value})


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("attr, value", [("max_iter", 1)])
@pytest.mark.parametrize(
    "model_name, quant_type, metric", [("FashionMNIST", "qat", binary_classification_metric)]
)
def test_update_valid_attr_method(attr, value, model_name, quant_type, metric):
    """Check that `_update_attr` can successfully update given valid attributes."""

    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    dataset = get_torchvision_dataset(DATASETS_ARGS[model_name], train_set=True)
    x_calib, y = data_calibration_processing(dataset, n_sample=1)

    search = BinarySearch(
        estimator=model,
        metric=metric,
    )

    search.run(x=x_calib, ground_truth=y, strategy=all, **{attr: value})


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
def test_non_convergence(model_class, parameters, load_data):
    """Check that binary search raises a user warning when convergence is not achieved."""

    x, y = load_data(model_class, **parameters)
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    model = instantiate_model_generic(model_class, n_bits=4)

    if is_model_class_in_a_list(model_class, get_sklearn_linear_models()):
        return

    metric = r2_score if is_regressor_or_partial_regressor(model) else binary_classification_metric

    search = BinarySearch(
        estimator=model,
        predict_method="predict",
        verbose=True,
        save=True,
        n_simulation=1,
        max_metric_loss=1e-9,
        delta_tolerance=1e-9,
        metric=metric,
        is_qat=False,
        max_iter=1,
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

    search = BinarySearch(estimator=model)
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

    search = BinarySearch(estimator=model)

    with pytest.raises(TypeError, match=".* is not valid."):
        search.eval_match(strategy, all_matches=numpy.random.choice([True, False], size=5))


@pytest.mark.parametrize("threshold", [0.019])
@pytest.mark.parametrize(
    "strategy",
    [
        all,
        lambda all_matches: numpy.median(all_matches) == 1,
    ],
)
@pytest.mark.parametrize(
    "model_name, quant_type", [("CustomModel", "qat"), ("CustomModel", "fp32")]
)
def test_binary_search_for_custom_models(model_name, quant_type, strategy, threshold):
    """Check if the returned `p_error` is valid for custom NNs."""

    # Load pretrained model
    model = load_torch_model(
        MODELS_ARGS[model_name][quant_type]["model_class"],
        MODELS_ARGS[model_name][quant_type]["path"],
        MODELS_ARGS[model_name][quant_type]["params"],
    )

    # Load data-set
    x, y = make_classification(**MODELS_ARGS[model_name]["dataset"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    search = BinarySearch(
        estimator=model,
        n_simulation=5,
        max_metric_loss=threshold,
        is_qat=quant_type == "qat",
        metric=top_k_accuracy_score,
        k=1,
        labels=numpy.arange(MODELS_ARGS[model_name]["dataset"]["n_classes"]),
    )

    largest_perror = search.run(x=x_calib, ground_truth=y, strategy=strategy)

    assert 1.0 > largest_perror > 0.0
    assert round(search.history[-1]["accuracy_difference"]) <= threshold


@pytest.mark.parametrize("threshold", [0.019])
@pytest.mark.parametrize(
    "strategy",
    [
        all,
        lambda all_matches: numpy.median(all_matches) == 1,
    ],
)
@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
@pytest.mark.parametrize("predict_method", ["predict", "predict_log_proba", "predict_proba"])
def test_binary_search_for_built_in_models(
    model_class, parameters, strategy, threshold, predict_method, load_data
):
    """Check if the returned `p_error` is valid for built-in models."""

    x, y = load_data(model_class, **parameters)
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    model = instantiate_model_generic(model_class, n_bits=4)

    if is_model_class_in_a_list(model_class, get_sklearn_linear_models()):
        return

    metric = r2_score if is_regressor_or_partial_regressor(model) else binary_classification_metric

    if hasattr(type(model), predict_method):

        search = BinarySearch(
            estimator=model,
            predict_method=predict_method,
            verbose=True,
            n_simulation=5,
            max_metric_loss=threshold,
            metric=metric,
            is_qat=False,
        )
    else:
        # The model does not have `predict_method`
        return

    largest_perror = search.run(x=x_calib, ground_truth=y, strategy=strategy)

    assert 1.0 > largest_perror > 0.0
    assert search.history[-1]["accuracy_difference"] <= threshold


@pytest.mark.parametrize("is_qat", [False, True])
def test_invalid_estimator(is_qat):
    """Check that binary search raises an exception for insupported models."""

    model = keras.Sequential(
        [
            keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dense(units=192, activation="relu"),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=10, activation="softmax"),
        ]
    )

    dataset = get_torchvision_dataset(DATASETS_ARGS["FashionMNIST"], train_set=True)
    x_calib, y = data_calibration_processing(dataset, n_sample=1)

    search = BinarySearch(
        estimator=model,
        is_qat=is_qat,
    )

    with pytest.raises(ValueError, match=".* is not supported. .*"):
        search.run(x=x_calib, ground_truth=y, strategy=all, max_iter=1)
