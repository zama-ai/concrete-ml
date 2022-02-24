import os
import time

import numpy as np
import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from concrete.ml.sklearn.linear_model import LogisticRegression
from concrete.ml.sklearn.qnn import NeuralNetClassifier
from concrete.ml.sklearn.tree import DecisionTreeClassifier

N_MAX_COMPILE_FHE = int(os.environ.get("N_MAX_COMPILE_FHE", 1000))
N_MAX_RUN_FHE = int(os.environ.get("N_MAX_RUN_FHE", 100))

datasets = [
    "credit-g",
    "blood-transfusion-service-center",
    "wilt",
    "tic-tac-toe",
    "kr-vs-kp",
    "qsar-biodeg",
    "wdbc",
    "steel-plates-fault",
    "diabetes",
    "ilpd",
    "phoneme",
    "spambase",
    "climate-model-simulation-crashes",
]

dataset_versions = {"wilt": 2}

classifiers = [DecisionTreeClassifier, NeuralNetClassifier, LogisticRegression]

benchmark_params = {
    # Benchmark different depths of the quantized decision tree
    DecisionTreeClassifier: [{"max_depth": 3}, {"max_depth": None}],
    LogisticRegression: [{"n_bits": 2}],
    NeuralNetClassifier: [
        # An FHE compatible config
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
        # for different accumulator bitwidths
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
            (3, 11),
            (4, 12),
            (5, 14),
            (6, 16),
            (7, 18),
            (8, 20),
            (9, 22),
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
        for n_b in range(2, 10)
    ],
}


def should_test_config_in_fhe(classifier, params, n_features):
    """Determine whether a benchmark config for a classifier should be tested in FHE"""

    # System override to disable FHE benchmarks (useful for debugging)
    if os.environ.get("BENCHMARK_NO_FHE", "0") == "1":
        return False

    if classifier is DecisionTreeClassifier:
        # Only small trees should be compiled to FHE
        if "max_depth" in params and params["max_depth"] is not None:
            return params["max_depth"] <= 3
        return False
    if classifier is NeuralNetClassifier:
        # For NNs only 7 bit accumulators with few neurons should be compiled to FHE
        return (
            params["module__n_accum_bits"] <= 7
            and params["module__n_hidden_neurons_multiplier"] == 1
        )
    if classifier is LogisticRegression:
        if params["n_bits"] <= 2 and n_features <= 14:
            return True

        if params["n_bits"] == 3 and n_features <= 2:
            return True

    raise ValueError(f"Classifier {str(classifier)} configurations not yet setup for FHE")


def run_and_report_metric(y_gt, y_pred, metric, metric_id, metric_label):
    """Run a single metric and report results to progress tracker"""
    value = metric(y_gt, y_pred) if y_gt.size > 0 else 0
    progress.measure(
        id=metric_id,
        label=metric_label,
        value=value,
    )


def run_and_report_all_metrics(y_gt, y_pred, metric_id_prefix, metric_label_prefix):
    """Run several metrics and report results to progress tracker with computed name and id"""

    metric_info = [
        (accuracy_score, "acc", "Accuracy"),
        (f1_score, "f1", "F1Score"),
        (matthews_corrcoef, "mcc", "MCC"),
        (average_precision_score, "ap", "AP"),
    ]
    for (metric, metric_id, metric_label) in metric_info:
        run_and_report_metric(
            y_gt,
            y_pred,
            metric,
            "_".join((metric_id_prefix, metric_id)),
            " ".join((metric_label_prefix, metric_label)),
        )


def train_and_test_on_dataset(classifier, dataset, config):
    """
    Train and test a classifier on a dataset

    This function trains a classifier type (caller must pass a class name) on an OpenML dataset
    identified by its name.
    """

    # Sometimes we want a specific version of a dataset, otherwise just get the 'active' one
    version = dataset_versions.get(dataset, "active")
    x_all, y_all = fetch_openml(
        name=dataset, version=version, as_frame=False, cache=True, return_X_y=True
    )

    # The OpenML datasets have target variables that might not be integers (for classification
    # integers would represent class ids). Mostly the targets are strings which we do not support.
    # We use an ordinal encoder to encode strings to integers
    if not y_all.dtype == np.int32:
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
    # FIXME: these parameters could be inferred from the data given to .fit
    # see https://github.com/zama-ai/concrete-ml-internal/issues/325

    if classifier is NeuralNetClassifier:
        classes = np.unique(y_all)
        config["module__input_dim"] = x_train.shape[1]
        config["module__n_outputs"] = len(classes)
        config["criterion__weight"] = compute_class_weight("balanced", classes=classes, y=y_train)

    concrete_classifier = classifier(**config)

    # Concrete ML classifiers follow the sklearn Estimator API but train differently than those
    # from sklearn. Our classifiers must work with quantized data or must determine data quantizers
    # after training the underlying sklearn classifier.
    # We call fit_benchmark to both fit our Concrete ML classifiers but also to return the sklearn
    # one that we would use if we were not using FHE. This classifier will be our baseline
    concrete_classifier, sklearn_classifier = concrete_classifier.fit_benchmark(x_train, y_train)

    # Predict with the sklearn classifier and compute accuracy. Although some datasets might be
    # imbalanced, we are not interested in the best metric for the case, but we want to measure
    # the difference in accuracy between the sklearn classifier and ours
    y_pred_sklearn = sklearn_classifier.predict(x_test)
    run_and_report_all_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    # Now predict with our classifier and report its accuracy
    y_pred_q = concrete_classifier.predict(x_test, execute_in_fhe=False)
    run_and_report_all_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

    n_features = x_train.shape[1] if x_train.ndim == 2 else 1

    if should_test_config_in_fhe(classifier, config, n_features):
        x_test_comp = x_test[0:N_MAX_COMPILE_FHE, :]

        # Compile and report compilation time
        t_start = time.time()
        concrete_classifier.compile(x_test_comp, compilation_configuration=BENCHMARK_CONFIGURATION)
        duration = time.time() - t_start
        progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

        # To keep the test short and to fit in RAM we limit the number of test samples
        x_test = x_test[0:N_MAX_RUN_FHE, :]
        y_test = y_test[0:N_MAX_RUN_FHE]

        # Now predict with our classifier and report its accuracy. We also measure
        # execution time per test sample
        t_start = time.time()
        y_pred_c = concrete_classifier.predict(x_test, execute_in_fhe=True)
        duration = time.time() - t_start

        run_and_report_all_metrics(y_test, y_pred_c, "fhe", "FHE")

        run_and_report_all_metrics(
            y_test, y_pred_q[0:N_MAX_RUN_FHE], "quant-clear-fhe-set", "Quantized Clear on FHE set"
        )

        progress.measure(
            id="fhe-inference_time",
            label="FHE Inference Time per sample",
            value=duration / x_test.shape[0] if x_test.shape[0] > 0 else 0,
        )


def benchmark_generator():
    for dataset in datasets:
        for classifier in classifiers:
            for config in benchmark_params[classifier]:
                yield (dataset, classifier, config)


def benchmark_name_generator(dataset, classifier, config, joiner):
    if classifier is DecisionTreeClassifier:
        if config["max_depth"] is not None:
            config_str = f"_{config['max_depth']}"
        else:
            config_str = ""
    elif classifier is NeuralNetClassifier:
        config_str = f"_{config['module__n_w_bits']}_{config['module__n_accum_bits']}"
    elif classifier is LogisticRegression:
        config_str = f"_{config['n_bits']}"
    return classifier.__name__ + config_str + joiner + dataset


# We run all the classifiers that we want to benchmark over all datasets listed
@progress.track(
    [
        {
            "id": benchmark_name_generator(dataset, classifier, config, "_"),
            "name": benchmark_name_generator(dataset, classifier, config, " on "),
            "parameters": {"classifier": classifier, "dataset": dataset, "config": config},
            "samples": 10,
        }
        for (dataset, classifier, config) in benchmark_generator()
    ]
)
def main(classifier, dataset, config):
    """
    This is the main test function called by the py-progress module. It just calls the
    benchmark function with the right parameter combination
    """
    train_and_test_on_dataset(classifier, dataset, config)
