import itertools
import os
import time

import numpy as np
import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from concrete.ml.sklearn.qnn import NeuralNetClassifier
from concrete.ml.sklearn.tree import DecisionTreeClassifier

# Set the number of benchmark samples to 1 to improve benchmark runtimes
# FIXME: use a better method to do this once it becomes available in py-progress
# See https://github.com/zama-ai/progress-tracker-python/issues/2
os.environ["PROGRESS_SAMPLES"] = "1"

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

classifiers = [DecisionTreeClassifier, NeuralNetClassifier]

benchmark_params = {
    DecisionTreeClassifier: {"max_depth": 3},
    NeuralNetClassifier: {
        "module__n_layers": 3,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": 7,
        "module__n_hidden_neurons_multiplier": 1,
        "max_epochs": 200,
        "verbose": 0,
    },
}


def train_and_test_on_dataset(classifier, dataset):
    """
    Train and test a classifier on a dataset

    This function trains a classifier type (caller must pass a class name) on an OpenML dataset
    identified by its name.
    """

    # Sometimes we want a specific version of a dataset, otherwise just get the 'active' one
    version = dataset_versions.get(dataset, "active")
    features, classes = fetch_openml(
        name=dataset, version=version, as_frame=False, cache=True, return_X_y=True
    )

    # The OpenML datasets have target variables that might not be integers (for classification
    # integers would represent class ids). Mostly the targets are strings which we do not support.
    # We use an ordinal encoder to encode strings to integers
    if not classes.dtype == np.int32:
        enc = OrdinalEncoder()
        classes = [[x] for x in classes]
        enc.fit(classes)
        classes = enc.transform(classes).astype(np.int64)
        classes = np.squeeze(classes)

    normalizer = StandardScaler()

    # Cast to a type that works for both sklearn and Torch
    features = features.astype(np.float32)

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        classes,
        test_size=0.15,
        random_state=42,
    )

    progress.measure(
        id="majority-class-percentage",
        label="Majority Class Percentage",
        value=np.max(np.bincount(y_test)) / y_test.size,
    )

    # Compute mean/stdev on training set and normalize both train and test sets with them
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Now instantiate the classifier, provide it with a custom configuration if we specify one
    # FIXME: these parameters could be infered from the data given to .fit
    # see https://github.com/zama-ai/concrete-ml-internal/issues/325
    config = benchmark_params.get(classifier, {})
    if classifier is NeuralNetClassifier:
        config["module__input_dim"] = x_train.shape[1]
        config["module__n_classes"] = len(np.unique(classes))

    concrete_classifier = classifier(**config)

    # Concrete ML classifiers follow the sklearn Estimator API but train differently than those
    # from sklearn. Our classifiers must work with quantized data or must determine data quantizers
    # after training the underlying sklearn classifier.
    # We call fit_benchmark to both fit our Concrete ML classifiers but also to return the sklearn
    # one that we would use if we were not using FHE. This classifier will be our baseline
    concrete_classifier, sklearn_classifer = concrete_classifier.fit_benchmark(x_train, y_train)

    # Predict with the sklearn classifier and compute accuracy. Although some datasets might be
    # imbalanced, we are not interested in the best metric for the case, but we want to measure
    # the difference in accuracy between the sklearn classifier and ours
    y_pred_sklearn = sklearn_classifer.predict(x_test)
    acc = accuracy_score(y_test, y_pred_sklearn)
    progress.measure(
        id="sklearn-acc",
        label="sklearn Accuracy",
        value=acc,
    )

    # Now predict with our classifier and report its accuracy
    y_pred_c = concrete_classifier.predict(x_test, execute_in_fhe=False)
    acc = accuracy_score(y_test, y_pred_c)
    progress.measure(
        id="quantized-clear-acc",
        label="Quantized Clear Accuracy",
        value=acc,
    )

    # Use only a subset of vectors for the bitwidth check
    # FIXME: once the HNP frontend is improved, increase this number to 1000
    # see: https://github.com/zama-ai/concrete-numpy-internal/issues/1374
    N_max_compile_fhe = 10
    if x_test.shape[0] > N_max_compile_fhe:
        x_test = x_test[0:N_max_compile_fhe, :]
        y_test = y_test[0:N_max_compile_fhe]

    # Compile and report compilation time
    t_start = time.time()
    concrete_classifier.compile(x_test, compilation_configuration=BENCHMARK_CONFIGURATION)
    duration = time.time() - t_start
    progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

    # To keep the test short and to fit in RAM we limit the number of test samples
    N_max_fhe = 10
    if x_test.shape[0] > N_max_fhe:
        x_test = x_test[0:N_max_fhe, :]
        y_test = y_test[0:N_max_fhe]

    # Now predict with our classifier and report its accuracy. We also measure execution time
    # per test sample
    t_start = time.time()
    y_pred_c = concrete_classifier.predict(x_test, execute_in_fhe=True)
    duration = time.time() - t_start

    # FIXME: accuracy is not informative for unbalanced data sets
    # see: https://github.com/zama-ai/concrete-ml-internal/issues/322
    acc = accuracy_score(y_test, y_pred_c)
    progress.measure(
        id="fhe-acc",
        label="FHE Accuracy",
        value=acc,
    )
    progress.measure(
        id="fhe-inference_time",
        label="FHE Inference Time per sample",
        value=duration / x_test.shape[0],
    )


# We run all the classifiers that we want to benchmark over all datasets listed
@progress.track(
    [
        {
            "id": classifier.__name__ + "_" + dataset,
            "name": classifier.__name__ + " on " + dataset,
            "parameters": {
                "classifier": classifier,
                "dataset": dataset,
            },
        }
        for (dataset, classifier) in itertools.product(datasets, classifiers)
    ]
)
def main(classifier, dataset):
    """
    This is the main test function called by the py-progress module. It just calls the
    benchmark function with the right parameter combination
    """
    train_and_test_on_dataset(classifier, dataset)
