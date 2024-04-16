import argparse
import inspect
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np
import py_progress_tracker as progress
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from concrete import fhe

try:
    from concrete.fhe.mlir.utils import MAXIMUM_TLU_BIT_WIDTH
except ImportError:  # For backward compatibility purposes
    try:
        from concrete.fhe.mlir.utils import (
            MAXIMUM_SIGNED_BIT_WIDTH_WITH_TLUS as MAXIMUM_TLU_BIT_WIDTH,
        )
    except ImportError:
        try:
            from concrete.fhe import MAXIMUM_TLU_BIT_WIDTH
        except ImportError:
            from concrete.fhe import MAXIMUM_BIT_WIDTH as MAXIMUM_TLU_BIT_WIDTH

# Hack to import all models currently implemented in Concrete ML
# (but that might not be implemented in targeted version)
# Since we can make no assumption about which models are
# imported and that one model not existing would cause the
# whole suite to crash we dynamically import our models

# Classification imports
CLASSIFIERS = []
CLASSIFIERS_NAMES = [
    "DecisionTreeClassifier",
    "LinearSVC",
    "LogisticRegression",
    "NeuralNetClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
]
for model_name in CLASSIFIERS_NAMES:
    try:
        model_class = getattr(__import__("concrete.ml.sklearn", fromlist=[model_name]), model_name)
        globals()[model_name] = model_class
        CLASSIFIERS.append(model_class)
    except (ImportError, AttributeError) as exception:
        print(exception)
CLASSIFIERS_STRING_TO_CLASS = {c.__name__: c for c in CLASSIFIERS}

# Regressors imports
REGRESSORS = []
REGRESSORS_NAMES = [
    "DecisionTreeRegressor",
    "LinearSVR",
    "LinearRegression",
    "ElasticNet",
    "Lasso",
    "Ridge",
    "NeuralNetRegressor",
    "RandomForestRegressor",
    "XGBRegressor",
    "SGDRegressor",
]
for model_name in REGRESSORS_NAMES:
    try:
        model_class = getattr(__import__("concrete.ml.sklearn", fromlist=[model_name]), model_name)
        globals()[model_name] = model_class
        REGRESSORS.append(model_class)
    except (ImportError, AttributeError) as exception:
        print(exception)
        print(f"model: {model_name} could not be imported.")
REGRESSORS_STRING_TO_CLASS = {c.__name__: c for c in REGRESSORS}

# GLMs imports
GLMS = []
GLMS_NAMES = [
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]
for model_name in GLMS_NAMES:
    try:
        model_class = getattr(__import__("concrete.ml.sklearn", fromlist=[model_name]), model_name)
        globals()[model_name] = model_class
        GLMS.append(model_class)
    except (ImportError, AttributeError) as exception:
        print(exception)
        print(f"model: {model_name} could not be imported.")
GLMS_STRING_TO_CLASS = {c.__name__: c for c in GLMS}

MODELS_STRING_TO_CLASS = {c.__name__: c for c in REGRESSORS + CLASSIFIERS + GLMS}

DEEP_LEARNING_NAMES = [
    "ShallowNarrowCNN",
    "Deep20CNN",
    "Deep50CNN",
    "Deep100CNN",
    "ShallowWideCNN",
]

MODEL_NAMES = CLASSIFIERS_NAMES + REGRESSORS_NAMES + GLMS_NAMES + DEEP_LEARNING_NAMES

NN_BENCHMARK_CONFIGS = (
    [
        # A FHE compatible config
        {
            "module__n_layers": 3,
            "module__n_w_bits": 2,
            "module__n_a_bits": 2,
            "module__n_accum_bits": 7,
            "module__n_hidden_neurons_multiplier": 1,
            "max_epochs": 200,
            "verbose": 0,
            "lr": 0.001,
        }
    ]
    + [
        # Pruned configurations that have approx. the same number of active neurons as the
        # FHE compatible config. This evaluates the accuracy that can be attained
        # for different accumulator bit-widths
        {
            "module__n_layers": 3,
            "module__n_w_bits": n_b,
            "module__n_a_bits": n_b,
            "module__n_accum_bits": n_b_acc,
            "module__n_hidden_neurons_multiplier": 4,
            "max_epochs": 200,
            "verbose": 0,
            "lr": 0.001,
        }
        for (n_b, n_b_acc) in [
            (2, 7),
            (10, 24),
        ]
    ]
    + [
        # Configs with all neurons active, to evaluate the accuracy of quantization of weights
        # and biases only
        {
            "module__n_layers": 3,
            "module__n_w_bits": n_b,
            "module__n_a_bits": n_b,
            "module__n_accum_bits": 32,
            "module__n_hidden_neurons_multiplier": 4,
            "max_epochs": 200,
            "verbose": 0,
            "lr": 0.001,
        }
        for n_b in [2, 3, 10]
    ]
)

LINEAR_REGRESSION_CONFIGS = [{"n_bits": n_bits} for n_bits in range(2, 21)]

XGB_CONFIGS = [
    {"max_depth": max_depth, "n_estimators": n_estimators, "n_bits": n_bits}
    for max_depth in [6]
    for n_estimators in [100]
    for n_bits in [2, 6]
]

RANDOM_FOREST_CONFIGS = [
    {"max_depth": max_depth, "n_estimators": n_estimators, "n_bits": n_bits}
    for max_depth in [10]
    for n_estimators in [100]
    for n_bits in [2, 6]
]

DECISION_TREE_CONFIGS = [
    {"max_depth": max_depth, "n_bits": n_bits} for max_depth in [5, 10] for n_bits in [2, 6]
]

DEEP_LEARNING_CONFIGS = [
    {"n_bits": n_bits, "p_error": p_error}
    for n_bits in [2, 5]
    for p_error in [2 ** (-40), 1 - 2 ** (-40)]
]

# Backward compatibility
if ("LinearRegression" in REGRESSORS) and (
    "use_sum_workaround"
    not in inspect.signature(REGRESSORS_STRING_TO_CLASS["LinearRegression"]).parameters
):
    LINEAR_REGRESSION_CONFIGS = [{"n_bits": n_bits} for n_bits in range(2, 11)]

BENCHMARK_PARAMS: Dict[str, List[Dict[str, Any]]] = {
    "XGBClassifier": XGB_CONFIGS,
    "XGBRegressor": XGB_CONFIGS,
    "RandomForestClassifier": RANDOM_FOREST_CONFIGS,
    "RandomForestRegressor": RANDOM_FOREST_CONFIGS,
    # Benchmark different depths of the quantized decision tree
    "DecisionTreeClassifier": DECISION_TREE_CONFIGS,
    "DecisionTreeRegressor": DECISION_TREE_CONFIGS,
    "LinearSVC": LINEAR_REGRESSION_CONFIGS,
    "LinearSVR": LINEAR_REGRESSION_CONFIGS,
    "LogisticRegression": LINEAR_REGRESSION_CONFIGS,
    "LinearRegression": LINEAR_REGRESSION_CONFIGS,
    "Lasso": LINEAR_REGRESSION_CONFIGS,
    "Ridge": LINEAR_REGRESSION_CONFIGS,
    "ElasticNet": LINEAR_REGRESSION_CONFIGS,
    "NeuralNetClassifier": NN_BENCHMARK_CONFIGS,
    "NeuralNetRegressor": NN_BENCHMARK_CONFIGS,
}

for deep_learning_name in DEEP_LEARNING_NAMES:
    BENCHMARK_PARAMS[deep_learning_name] = DEEP_LEARNING_CONFIGS

REGRESSION_DATASETS = [
    "one-hundred-plants-margin",
]
CLASSIFICATION_DATASETS = ["CreditCardFraudDetection"]
DATASET_VERSIONS = {
    "wilt": 2,
    "abalone": 5,
    "us_crime": 2,
    "Brazilian_houses": 4,
    "Moneyball": 2,
    "Yolanda": 2,
    "quake": 2,
    "house_sales": 3,
}


# This is only for benchmarks to speed up compilation times
# Jit compiler is now deprecated and will soon be removed, it is thus forced to False by default
# Parameter `enable_unsafe_features` and `use_insecure_key_cache` are needed in order to be able to
# cache generated keys through `insecure_key_cache_location`. As the name suggests, these
# parameters are unsafe and should only be used for debugging in development
BENCHMARK_CONFIGURATION = fhe.Configuration(
    dump_artifacts_on_unexpected_failures=True,
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="ConcreteNumpyKeyCache",
)


def run_and_report_metric(y_gt, y_pred, metric, metric_id, metric_label):
    """Run a single metric and report results to progress tracker"""
    value = metric(y_gt, y_pred) if y_gt.size > 0 else 0
    progress.measure(
        id=metric_id,
        label=metric_label,
        value=value,
    )


def run_and_report_classification_metrics(
    y_gt, y_pred, metric_id_prefix, metric_label_prefix, use_f1=True
):
    """Run several metrics and report results to progress tracker with computed name and id"""

    metric_info = [
        (accuracy_score, "acc", "Accuracy"),
        (matthews_corrcoef, "mcc", "MCC"),
    ]

    if use_f1:
        metric_info += [
            (f1_score, "f1", "F1Score"),
        ]

    for metric, metric_id, metric_label in metric_info:
        run_and_report_metric(
            y_gt,
            y_pred,
            metric,
            "_".join((metric_id_prefix, metric_id)),
            " ".join((metric_label_prefix, metric_label)),
        )


def run_and_report_regression_metrics(y_gt, y_pred, metric_id_prefix, metric_label_prefix):
    """Run several metrics and report results to progress tracker with computed name and id"""

    metric_info = [(r2_score, "r2_score", "R2Score"), (mean_squared_error, "MSE", "MSE")]
    for metric, metric_id, metric_label in metric_info:
        run_and_report_metric(
            y_gt,
            y_pred,
            metric,
            "_".join((metric_id_prefix, metric_id)),
            " ".join((metric_label_prefix, metric_label)),
        )


def seed_everything(seed):
    random.seed(seed)
    seed += 1
    np.random.seed(seed % 2**32)
    seed += 1
    torch.manual_seed(seed)
    seed += 1
    torch.use_deterministic_algorithms(True)
    return seed


# pylint: disable-next=too-many-return-statements, too-many-branches, redefined-outer-name
def should_test_config_in_fhe(
    model: type, config: Dict[str, Any], local_args: argparse.Namespace
) -> bool:
    """Determine whether a benchmark config for a classifier should be tested in FHE"""
    if local_args.execute_in_fhe != "auto":
        return local_args.execute_in_fhe

    model_name = model.__name__
    assert config is not None

    # System override to disable FHE benchmarks (useful for debugging)
    if os.environ.get("BENCHMARK_NO_FHE", "0") == "1":
        return False

    if model_name in {"DecisionTreeClassifier", "DecisionTreeRegressor"}:
        # Only small trees should be compiled to FHE
        if "max_depth" in config and config["max_depth"] is not None and config["n_bits"] <= 7:
            return True
        return False

    if model_name in {
        "LogisticRegression",
        "Lasso",
        "ElasticNet",
        "Ridge",
        "LinearSVC",
        "LinearSVR",
        "LinearRegression",
        "TweedieRegressor",
        "PoissonRegressor",
        "GammaRegressor",
        "SGDRegressor",
    }:
        return True

    if model_name in {
        "XGBClassifier",
        "XGBRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
    }:
        if config["n_bits"] <= 7:
            return True
        return False

    if model_name in {"NeuralNetRegressor", "NeuralNetClassifier"}:
        # For NNs only 7 bit accumulators with few neurons should be compiled to FHE
        return (
            config["module__n_accum_bits"] <= 7
            and config["module__n_hidden_neurons_multiplier"] == 1
        )

    raise ValueError(f"{str(model_name)} is not setup for FHE yet.")


# pylint: disable-next=too-many-branches
def train_and_test_regressor(
    regressor: type, dataset: str, config: Dict[str, Any], local_args: argparse.Namespace
):
    # Could be changed but not very useful
    size_of_compilation_dataset = 1000

    if local_args.verbose:
        print("Start")
        time_current = time.time()

    version = DATASET_VERSIONS.get(dataset, "active")
    X, y = fetch_openml(name=dataset, version=version, as_frame=False, cache=True, return_X_y=True)
    if y.ndim == 1:
        y = np.expand_dims(y, 1)

    if regressor.__name__ == "NeuralNetRegressor":
        # Cast to a type that works for both sklearn and Torch
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    # Split it into train/test and sort the sets for nicer visualization
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    if regressor.__name__ == "NeuralNetRegressor":
        normalizer = StandardScaler()
        # Compute mean/stdev on training set and normalize both train and test sets with them
        x_train = normalizer.fit_transform(x_train)
        x_test = normalizer.transform(x_test)

    concrete_regressor = regressor(**config)

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Fit")

    # We call fit_benchmark to both fit our Concrete ML regressors but also to return the sklearn
    # one that we would use if we were not using FHE. This regressor will be our baseline
    concrete_regressor, sklearn_regressor = concrete_regressor.fit_benchmark(x_train, y_train)

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict with scikit-learn")

    # Predict with the sklearn regressor and compute goodness of fit
    y_pred_sklearn = sklearn_regressor.predict(x_test)
    run_and_report_regression_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict in clear")

    # Now predict with our regressor and report its goodness of fit
    y_pred_q = concrete_regressor.predict(x_test, fhe="disable")
    run_and_report_regression_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

    if should_test_config_in_fhe(regressor, config, local_args):

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Compile")

        x_test_comp = x_test[0:size_of_compilation_dataset, :]

        # Compile and report compilation time
        t_start = time.time()
        fhe_circuit = concrete_regressor.compile(x_test_comp, configuration=BENCHMARK_CONFIGURATION)

        # Dump MLIR
        if local_args.mlir_only:
            mlirs_dir: Path = Path(__file__).parents[1] / "MLIRs"
            benchmark_name = benchmark_name_generator(
                dataset, concrete_regressor.__class__, config, "_"
            )
            mlirs_dir.mkdir(parents=True, exist_ok=True)
            with open(mlirs_dir / f"{benchmark_name}.mlir", "w", encoding="utf-8") as file:
                file.write(fhe_circuit.mlir)
            return

        duration = time.time() - t_start
        progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Key generation")

        t_start = time.time()
        fhe_circuit.keygen()
        duration = time.time() - t_start
        progress.measure(id="fhe-keygen-time", label="FHE Key Generation Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print(f"Predict in FHE ({local_args.fhe_samples} samples)")

        # To keep the test short and to fit in RAM we limit the number of test samples
        x_test = x_test[0 : local_args.fhe_samples, :]
        y_test = y_test[0 : local_args.fhe_samples]

        # Now predict with our regressor and report its goodness of fit. We also measure
        # execution time per test sample
        t_start = time.time()
        y_pred_c = concrete_regressor.predict(x_test, fhe="execute")
        duration = time.time() - t_start

        run_and_report_regression_metrics(y_test, y_pred_c, "fhe", "FHE")

        run_and_report_regression_metrics(
            y_test,
            y_pred_q[0 : local_args.fhe_samples],
            "quant-clear-fhe-set",
            "Quantized Clear on FHE set",
        )

        progress.measure(
            id="fhe-inference_time",
            label="FHE Inference Time per sample",
            value=duration / x_test.shape[0] if x_test.shape[0] > 0 else 0,
        )

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("End")


def train_and_test_classifier(
    classifier: type, dataset: str, config: Dict[str, Any], local_args: argparse.Namespace
):
    """
    Train and test a classifier on a data-set

    This function trains a classifier type (caller must pass a class name) on an OpenML data-set
    identified by its name.
    """

    # Could be changed but not very useful
    size_of_compilation_dataset = 1000

    if local_args.verbose:
        print("Start")
        time_current = time.time()

    # Sometimes we want a specific version of a data-set, otherwise just get the 'active' one
    version = DATASET_VERSIONS.get(dataset, "active")
    x_all, y_all = fetch_openml(
        name=dataset, version=version, as_frame=False, cache=True, return_X_y=True
    )

    # The OpenML data-sets have target variables that might not be integers (for classification
    # integers would represent class ids). Mostly the targets are strings which we do not support.
    # We use an ordinal encoder to encode strings to integers
    if not y_all.dtype == np.int64:
        enc = OrdinalEncoder()
        y_all = [[y] for y in y_all]
        enc.fit(y_all)
        y_all = enc.transform(y_all).astype(np.int64)
        y_all = np.squeeze(y_all)

    normalizer = StandardScaler()

    # Cast to a type that works for both sklearn and Torch
    x_all = x_all.astype(np.float32)

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.15, random_state=42, shuffle=True, stratify=y_all
    )

    pct_pos_test = np.max(np.bincount(y_test)) / y_test.size
    progress.measure(
        id="majority-class-percentage",
        label="Majority Class Percentage",
        value=pct_pos_test,
    )

    # Compute mean/stdev on training set and normalize both train and test sets with them
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Now instantiate the classifier, provide it with a custom configuration if we specify one
    # These parameters could be inferred from the data given to .fit

    if classifier.__name__ == "NeuralNetClassifier":
        classes = np.unique(y_all)
        config["criterion__weight"] = compute_class_weight("balanced", classes=classes, y=y_train)

    concrete_classifier = classifier(**config)

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Fit")

    # Concrete ML classifiers follow the sklearn Estimator API but train differently than those
    # from sklearn. Our classifiers must work with quantized data or must determine data quantizers
    # after training the underlying sklearn classifier.
    # We call fit_benchmark to both fit our Concrete ML classifiers but also to return the sklearn
    # one that we would use if we were not using FHE. This classifier will be our baseline
    concrete_classifier, sklearn_classifier = concrete_classifier.fit_benchmark(x_train, y_train)

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict with scikit-learn")

    # Predict with the sklearn classifier and compute accuracy. Although some data-sets might be
    # imbalanced, we are not interested in the best metric for the case, but we want to measure
    # the difference in accuracy between the sklearn classifier and ours
    y_pred_sklearn = sklearn_classifier.predict(x_test)
    run_and_report_classification_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict in clear")

    # Now predict with our classifier and report its accuracy
    y_pred_q = concrete_classifier.predict(x_test, fhe="disable")
    run_and_report_classification_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

    if should_test_config_in_fhe(classifier, config, local_args):

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Compile")

        x_test_comp = x_test[0:size_of_compilation_dataset, :]

        # Compile and report compilation time
        t_start = time.time()
        fhe_circuit = concrete_classifier.compile(
            x_test_comp, configuration=BENCHMARK_CONFIGURATION
        )

        # Dump MLIR
        if local_args.mlir_only:
            mlirs_dir: Path = Path(__file__).parents[1] / "MLIRs"
            benchmark_name = benchmark_name_generator(
                dataset, concrete_classifier.__class__, config, "_"
            )
            mlirs_dir.mkdir(parents=True, exist_ok=True)
            with open(mlirs_dir / f"{benchmark_name}.mlir", "w", encoding="utf-8") as file:
                file.write(fhe_circuit.mlir)
            return

        duration = time.time() - t_start
        progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Key generation")

        t_start = time.time()
        fhe_circuit.keygen()
        duration = time.time() - t_start
        progress.measure(id="fhe-keygen-time", label="FHE Key Generation Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print(f"Predict in FHE ({local_args.fhe_samples} samples)")

        # To keep the test short and to fit in RAM we limit the number of test samples
        x_test = x_test[0 : local_args.fhe_samples, :]
        y_test = y_test[0 : local_args.fhe_samples]

        # Now predict with our classifier and report its accuracy. We also measure
        # execution time per test sample
        t_start = time.time()
        y_pred_c = concrete_classifier.predict(x_test, fhe="execute")
        duration = time.time() - t_start

        run_and_report_classification_metrics(y_test, y_pred_c, "fhe", "FHE")

        run_and_report_classification_metrics(
            y_test,
            y_pred_q[0 : local_args.fhe_samples],
            "quant-clear-fhe-set",
            "Quantized Clear on FHE set",
        )

        progress.measure(
            id="fhe-inference_time",
            label="FHE Inference Time per sample",
            value=duration / x_test.shape[0] if x_test.shape[0] > 0 else 0,
        )

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("End")


# pylint: disable-next=redefined-outer-name
def benchmark_generator(local_args) -> Iterator[Tuple[str, type, Dict[str, Any]]]:
    """Generates all elements to test."""
    for dataset in local_args.datasets:
        for model_class_ in local_args.models:
            if local_args.configs is None:
                for config in BENCHMARK_PARAMS[model_class_.__name__]:
                    yield (dataset, model_class_, config)
            else:
                for config in local_args.configs:
                    yield (dataset, model_class_, config)


# For GLMs
def compute_number_of_components(n_bits: Union[Dict, int]) -> int:
    """Computes the maximum number of PCA components possible for executing a model in FHE."""
    if isinstance(n_bits, int):
        n_bits_inputs = n_bits
        n_bits_weights = n_bits
    else:
        n_bits_inputs = n_bits["op_inputs"]
        n_bits_weights = n_bits["op_weights"]

    n_components = math.floor(
        (2**MAXIMUM_TLU_BIT_WIDTH - 1) / ((2**n_bits_inputs - 1) * (2**n_bits_weights - 1))
    )
    return n_components


# pylint: disable-next=redefined-outer-name,too-many-branches
def benchmark_name_generator(
    dataset_name: str, model: type, config: Dict[str, Any], joiner: str = "_"
) -> str:
    """Turns a combination of data-set + model + hyper-parameters and returns a string"""
    assert isinstance(model, type), f"Wrong type: {type(model)} - {model}"

    model_name = model.__name__

    assert (
        model_name in MODEL_NAMES
    ), f"Wrong model name. Expected one of {MODEL_NAMES}, got {model_name}"

    if model_name in {
        "LinearSVR",
        "LinearSVC",
        "LogisticRegression",
        "LinearRegression",
        "Lasso",
        "ElasticNet",
        "Ridge",
    }:
        config_str = f"_{config['n_bits']}"

    elif model_name == "NeuralNetRegressor":
        config_str = f"_{config['module__n_a_bits']}_{config['module__n_accum_bits']}"

    elif model_name == "NeuralNetClassifier":
        config_str = f"_{config['module__n_w_bits']}_{config['module__n_accum_bits']}"

    elif model_name in {"DecisionTreeRegressor", "DecisionTreeClassifier"}:
        if config["max_depth"] is not None:
            config_str = f"_{config['max_depth']}_{config['n_bits']}"
        else:
            config_str = f"_{config['n_bits']}"

    elif model_name in {"XGBClassifier", "XGBRegressor"}:
        if config["max_depth"] is not None:
            config_str = f"_{config['max_depth']}_{config['n_estimators']}_{config['n_bits']}"
        else:
            config_str = f"_{config['n_estimators']}_{config['n_bits']}"

    elif model_name in {"RandomForestRegressor", "RandomForestClassifier"}:
        if config["max_depth"] is not None:
            config_str = f"_{config['max_depth']}_{config['n_estimators']}_{config['n_bits']}"
        else:
            config_str = f"_{config['n_estimators']}_{config['n_bits']}"

    # GLMs
    elif model_name in GLMS_NAMES:
        if isinstance(config["n_bits"], int):
            n_bits_inputs = config["n_bits"]
            n_bits_weights = config["n_bits"]
        else:
            n_bits_inputs = config["n_bits"]["op_inputs"]
            n_bits_weights = config["n_bits"]["op_weights"]

        pca_n_components = compute_number_of_components(config["n_bits"])
        config_str = f"_{n_bits_inputs}_{n_bits_weights}_{pca_n_components}"

    elif model_name in DEEP_LEARNING_NAMES:
        config_str = f"_{config['n_bits']}"

    # We remove underscores to make sure to not have any conflict when splitting
    return model_name.replace("_", "-") + config_str + joiner + dataset_name.replace("_", "-")


# Add tests:
# - Bijection between both functions
# - The functions support all models
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1866


# pylint: disable-next=too-many-branches, redefined-outer-name
def benchmark_name_to_config(
    benchmark_name: str, joiner: str = "_"
) -> Tuple[str, str, Dict[str, Any]]:
    """Convert a benchmark name to each part"""
    splitted = benchmark_name.split(joiner)
    model_name = splitted[0]
    dataset_name = splitted[-1]
    config_str = splitted[1:-1]
    config_dict = {}

    if model_name in {
        "LinearRegression",
        "LinearSVR",
        "Lasso",
        "ElasticNet",
        "Ridge",
        "LinearSVC",
        "LogisticRegression",
    }:
        config_dict["n_bits"] = int(config_str[0])

    elif model_name == "NeuralNetRegressor":
        config_dict["module__n_a_bits"] = int(config_str[0])
        config_dict["module__n_accum_bits"] = int(config_str[1])

    elif "model_name" == "NeuralNetClassifier":
        config_dict["module__n_w_bits"] = int(config_str[0])
        config_dict["module__n_accum_bits"] = int(config_str[1])

    elif model_name in {"DecisionTreeClassifier", "DecisionTreeRegressor"}:
        if len(config_str) == 2:
            config_dict["max_depth"] = int(config_str[0])
            config_dict["n_bits"] = int(config_str[1])
        elif len(config_str) == 1:
            config_dict["n_bits"] = int(config_str[0])
        else:
            raise ValueError(
                f"{benchmark_name} couldn't be parsed\n"
                f"{config_str} does not match any know configuration"
            )

    elif model_name in {"XGBClassifier", "XGBRegressor"}:
        if len(config_str) == 3:
            config_dict["max_depth"] = int(config_str[0])
            config_dict["n_estimators"] = int(config_str[1])
            config_dict["n_bits"] = int(config_str[2])
        elif len(config_str) == 2:
            config_dict["n_estimators"] = int(config_str[0])
            config_dict["n_bits"] = int(config_str[1])
        else:
            raise ValueError(
                f"{benchmark_name} couldn't be parsed\n"
                f"{config_str} does not match any know configuration"
            )

    elif model_name in {"RandomForestClassifier", "RandomForestRegressor"}:
        if len(config_str) == 3:
            config_dict["max_depth"] = int(config_str[0])
            config_dict["n_estimators"] = int(config_str[1])
            config_dict["n_bits"] = int(config_str[2])
        elif len(config_str) == 2:
            config_dict["n_estimators"] = int(config_str[0])
            config_dict["n_bits"] = int(config_str[1])
        else:
            raise ValueError(
                f"{benchmark_name} couldn't be parsed\n"
                f"{config_str} does not match any know configuration"
            )
    # GLMs
    elif model_name in {"PoissonRegressor", "GammaRegressor", "TweedieRegressor"}:
        if len(config_str) == 3:
            config_dict["n_bits_inputs"] = int(config_str[0])
            config_dict["n_bits_weights"] = int(config_str[1])
            config_dict["pca_n_components"] = int(config_str[2])
        else:
            raise ValueError(
                f"{benchmark_name} couldn't be parsed\n"
                f"{config_str} does not match any know configuration"
            )

    return model_name, dataset_name, config_dict
