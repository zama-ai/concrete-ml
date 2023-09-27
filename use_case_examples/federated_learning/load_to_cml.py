import pickle
from pathlib import Path

import federated_utils
import numpy

from concrete.ml.sklearn import LogisticRegression

if __name__ == "__main__":
    # Load scikit-learn model
    path_to_model = Path("./model.pkl")
    if not path_to_model.exists() or not path_to_model.is_file():
        raise ValueError(
            "Couldn't find file 'model.pkl'.\n"
            "Please run 'run.sh' to train the model with federated learning."
        )
    with path_to_model.open("rb") as file:
        sklearn_model = pickle.load(file)

    # Compile model without data since the server doesn't have access to it
    # Indeed in this scenario the users have the data but the server doesn't.
    # We then have to compile the model using random input sampled with the same
    # low and high bounds as the real data, in this context [0, 255].
    number_of_compile_samples = 1000
    compile_set = numpy.random.randint(0, 255, (number_of_compile_samples, 784)).astype(float)
    sklearn_model.classes_ = sklearn_model.classes_.astype(int)
    model = LogisticRegression.from_sklearn_model(sklearn_model, compile_set)
    model.compile(compile_set)

    # Evaluate the model
    # Load MNIST dataset from https://www.openml.org/d/554
    (_, _), (X_test, y_test) = federated_utils.load_mnist()

    concrete_predictions = model.predict(X_test)
    sklearn_predictions = sklearn_model.predict(X_test)
    simulation_predictions = model.predict(X_test, fhe="simulate")

    print(
        (concrete_predictions == sklearn_predictions).sum() / len(y_test),
        "quantized vs scikit-learn",
    )
    print(
        (concrete_predictions == simulation_predictions).sum() / len(y_test),
        "quantized vs simulation",
    )
    print(
        (simulation_predictions == sklearn_predictions).sum() / len(y_test),
        "simulation vs scikit-learn",
    )

    print((concrete_predictions == y_test).sum() / len(y_test), "quantized accuracy")
    print((sklearn_predictions == y_test).sum() / len(y_test), "scikit-learn accuracy")
    print((simulation_predictions == y_test).sum() / len(y_test), "simulation accuracy")
