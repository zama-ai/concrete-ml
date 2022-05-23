import argparse
import json
import os
import random
import time

import numpy as np
import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION, run_and_report_classification_metrics, seed_everything
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    LinearSVC,
    LogisticRegression,
    NeuralNetClassifier,
    RandomForestClassifier,
    XGBClassifier,
)

possible_datasets = [
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

possible_classifiers = [
    RandomForestClassifier,
    XGBClassifier,
    DecisionTreeClassifier,
    NeuralNetClassifier,
    LogisticRegression,
    LinearSVC,
]

classifiers_string_to_class = {c.__name__: c for c in possible_classifiers}

benchmark_params = {
    RandomForestClassifier: [
        {"max_depth": max_depth, "n_estimators": n_estimators, "n_bits": n_bits}
        for max_depth in [15]
        for n_estimators in [100]
        for n_bits in [7, 16]
    ],
    XGBClassifier: [
        {"max_depth": max_depth, "n_estimators": n_estimators, "n_bits": n_bits}
        for max_depth in [7]
        for n_estimators in [50]
        for n_bits in [7, 16]
    ],
    # Benchmark different depths of the quantized decision tree
    DecisionTreeClassifier: [{"max_depth": 3}, {"max_depth": None}],
    LinearSVC: [{"n_bits": 2}],
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


# pylint: disable=too-many-return-statements, too-many-branches
def should_test_config_in_fhe(classifier, params, n_features, local_args):
    """Determine whether a benchmark config for a classifier should be tested in FHE"""

    if local_args.execute_in_fhe != "auto":
        return local_args.execute_in_fhe

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

    if classifier is LinearSVC:
        if params["n_bits"] <= 2 and n_features <= 14:
            return True

        if params["n_bits"] == 3 and n_features <= 2:
            return True

    if classifier is XGBClassifier or classifier is RandomForestClassifier:
        if params["n_bits"] <= 7:
            return True
        return False

    raise ValueError(f"Classifier {str(classifier)} configurations not yet setup for FHE")


# pylint: enable=too-many-return-statements, too-many-branches


def train_and_test_on_dataset(classifier, dataset, config, local_args):
    """
    Train and test a classifier on a dataset

    This function trains a classifier type (caller must pass a class name) on an OpenML dataset
    identified by its name.
    """

    # Could be changed but not very useful
    size_of_compilation_dataset = 1000

    if local_args.verbose:
        print("Start")
        time_current = time.time()

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

    # Predict with the sklearn classifier and compute accuracy. Although some datasets might be
    # imbalanced, we are not interested in the best metric for the case, but we want to measure
    # the difference in accuracy between the sklearn classifier and ours
    y_pred_sklearn = sklearn_classifier.predict(x_test)
    run_and_report_classification_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict in clear")

    # Now predict with our classifier and report its accuracy
    y_pred_q = concrete_classifier.predict(x_test, execute_in_fhe=False)
    run_and_report_classification_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

    n_features = x_train.shape[1] if x_train.ndim == 2 else 1

    if should_test_config_in_fhe(classifier, config, n_features, local_args):

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Compile")

        x_test_comp = x_test[0:size_of_compilation_dataset, :]

        # Compile and report compilation time
        t_start = time.time()
        forward_fhe = concrete_classifier.compile(
            x_test_comp, configuration=BENCHMARK_CONFIGURATION
        )
        duration = time.time() - t_start
        progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Key generation")

        t_start = time.time()
        forward_fhe.keygen()
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
        y_pred_c = concrete_classifier.predict(x_test, execute_in_fhe=True)
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


def benchmark_generator(local_args):
    for dataset in local_args.datasets:
        for classifier in local_args.classifiers:
            if local_args.configs is None:
                for config in benchmark_params[classifier]:
                    yield (dataset, classifier, config)
            else:
                for config in local_args.configs:
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
    elif classifier is LinearSVC:
        config_str = f"_{config['n_bits']}"
    elif classifier is XGBClassifier:
        if config["max_depth"] is not None:
            config_str = f"_{config['max_depth']}_{config['n_estimators']}_{config['n_bits']}"
        else:
            config_str = f"_{config['n_estimators']}_{config['n_bits']}"
    elif classifier is RandomForestClassifier:
        if config["max_depth"] is not None:
            config_str = f"_{config['max_depth']}_{config['n_estimators']}_{config['n_bits']}"
        else:
            config_str = f"_{config['n_estimators']}_{config['n_bits']}"
    return classifier.__name__ + config_str + joiner + dataset


def argument_manager():
    # Manage arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="show more information on stdio")
    parser.add_argument(
        "--datasets",
        choices=possible_datasets,
        type=str,
        nargs="+",
        default=None,
        help="dataset(s) to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="set the seed for reproducibility",
    )
    parser.add_argument(
        "--classifiers",
        choices=classifiers_string_to_class.keys(),
        nargs="+",
        default=None,
        help="classifier(s) to use",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=json.loads,
        default=None,
        help="config(s) to use",
    )
    parser.add_argument(
        "--model_samples",
        type=int,
        default=1,
        help="number of model samples (ie, overwrite PROGRESS_SAMPLES)",
    )
    parser.add_argument(
        "--fhe_samples", type=int, default=1, help="number of FHE samples on which to predict"
    )
    parser.add_argument(
        "--execute_in_fhe",
        action="store_true",
        default="auto",
        help="force to execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    parser.add_argument(
        "--dont_execute_in_fhe",
        action="store_true",
        help="force to not execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    args, _ = parser.parse_known_args()
    if args.dont_execute_in_fhe:
        assert args.execute_in_fhe == "auto"
        args.execute_in_fhe = False

    if args.datasets is None:
        args.datasets = possible_datasets
    if args.classifiers is None:
        args.classifiers = possible_classifiers
    else:
        args.classifiers = [classifiers_string_to_class[c] for c in args.classifiers]

    return args


def main():

    # Parameters by the user
    args = argument_manager()

    print(f"Will perform benchmarks on {len(list(benchmark_generator(args)))} test cases")
    print(f"Using --seed {args.seed}")

    # Seed everything we can
    seed_everything(args.seed)

    # We run all the classifiers that we want to benchmark over all datasets listed
    @progress.track(
        [
            {
                "id": benchmark_name_generator(dataset, classifier, config, "_"),
                "name": benchmark_name_generator(dataset, classifier, config, " on "),
                "parameters": {"classifier": classifier, "dataset": dataset, "config": config},
                "samples": args.model_samples,
            }
            for (dataset, classifier, config) in benchmark_generator(args)
        ]
    )
    def perform_benchmark(classifier, dataset, config):
        """
        This is the test function called by the py-progress module. It just calls the
        benchmark function with the right parameter combination
        """
        train_and_test_on_dataset(classifier, dataset, config, args)


if __name__ == "__main__":
    main()
