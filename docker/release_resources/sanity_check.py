import random

# Check that concrete-numpy extra packages are installed in the docker image
import pygraphviz

print("pygraphviz import check OK")

import concrete.numpy as hnp
import numpy
from concrete.common.compilation import CompilationConfiguration
from sklearn.datasets import fetch_openml
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

from concrete import ml
from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDecisionTreeClassifier


def ml_check():
    print(ml.__version__)

    features, classes = fetch_openml(data_id=44, as_frame=False, cache=True, return_X_y=True)
    classes = classes.astype(numpy.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        classes,
        test_size=0.15,
        random_state=42,
    )

    model = ConcreteDecisionTreeClassifier()
    model.fit(x_train, y_train)

    # Compute average precision on test

    y_pred = model.predict(x_test)
    average_precision = average_precision_score(y_test, y_pred)
    print(f"Average precision-recall score: {average_precision:0.2f}")

    y_pred = model.predict(x_test)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, y_pred, normalize="true"
    ).ravel()

    num_samples = len(y_test)
    num_spam = sum(y_test)

    print(f"Number of test samples: {num_samples}")
    print(f"Number of spams in test samples: {num_spam}")

    print(f"True Negative (legit mail well classified) rate: {true_negative}")
    print(f"False Positive (legit mail classified as spam) rate: {false_positive}")
    print(f"False Negative (spam mail classified as legit) rate: {false_negative}")
    print(f"True Positive (spam well classified) rate: {true_positive}")

    # We first compile the model with some data, here the training set
    model.compile(
        x_train,
        compilation_configuration=CompilationConfiguration(enable_unsafe_features=True),
        use_virtual_lib=True,
    )

    # Predict in VL for a few examples
    y_pred_vl = model.predict(x_test[:10], execute_in_fhe=True)

    # Check prediction VL vs sklearn
    print(f"Prediction VL:      {y_pred_vl}")
    print(f"Prediction sklearn: {y_pred[:10]}")

    print(
        f"{numpy.sum(y_pred_vl==y_pred[:10])}/10 "
        "predictions are similar between the VL model and the clear sklearn model."
    )


def cn_check():
    def function_to_compile(x):
        return x + 42

    n_bits = 3

    compiler = hnp.NPFHECompiler(
        function_to_compile,
        {"x": "encrypted"},
    )

    print("Compiling...")

    engine = compiler.compile_on_inputset(range(2**n_bits))

    inputs = []
    labels = []
    for _ in range(4):
        sample_x = random.randint(0, 2**n_bits - 1)

        inputs.append([sample_x])
        labels.append(function_to_compile(*inputs[-1]))

    correct = 0
    for idx, (input_i, label_i) in enumerate(zip(inputs, labels), 1):
        print(f"Inference #{idx}")
        result_i = engine.run(*input_i)

        if result_i == label_i:
            correct += 1

    print(f"{correct}/{len(inputs)}")


def main():
    ml_check()
    cn_check()


if __name__ == "__main__":
    main()
