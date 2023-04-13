import numpy
from sklearn.datasets import load_breast_cancer

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier

if __name__ == "__main__":
    # First get some data and train a model.
    X, y = load_breast_cancer(return_X_y=True)

    assert isinstance(X, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)

    # Split X into X_model_owner and X_client
    X_train = X[:-10]
    y_train = y[:-10]

    # Train the model and compile it
    model = XGBClassifier(n_bits=2, n_estimators=8, max_depth=3)
    model.fit(X_train, y_train)
    model.compile(X_train)
    dev = FHEModelDev("./dev", model)
    dev.save()
