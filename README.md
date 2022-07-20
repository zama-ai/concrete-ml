# Concrete ML

**Concrete-ML** is an open-source set of tools which aims to simplify the use of fully homomorphic encryption (FHE) for data scientists. Particular care was given to the simplicity of our Python package in order to make it usable by any data scientist, even those without prior cryptography knowledge. Notably, our APIs are as close as possible to scikit-learn and torch APIs to simplify adoption by our users.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Links](#links)
- [For end users](#for-end-users)
  - [Installation.](#installation)
  - [Supported models.](#supported-models)
  - [Simple ML examples with scikit-learn.](#simple-ml-examples-with-scikit-learn)
- [For developers](#for-developers)
  - [Project setup.](#project-setup)
  - [Documenting.](#documenting)
  - [Developing.](#developing)
  - [Contributing.](#contributing)
- [License](#license)

<!-- mdformat-toc end -->

## Links

- [documentation](https://docs.zama.ai/concrete-ml/)
- [community website](https://community.zama.ai/c/concrete-ml)
- [demos](https://docs.zama.ai/concrete-ml/advanced-examples/advanced_examples)

## For end users

### Installation.

The preferred way to use **Concrete-ML** is through Docker. You can get our Docker image by pulling the latest Docker image:

`docker pull zamafhe/concrete-ml:latest`

To install **Concrete-ML** from PyPi, run the following:

`pip install concrete-ml`

You can find more detailed installation instructions in [installing.md](docs/user/basics/installing.md)

### Supported models.

Here is a list of ML algorithms currently supported in this library:

- LinearRegression (sklearn)
- LogisticRegression (sklearn)
- SVM - SVC and SVR (sklearn)
- DecisionTreeClassifier (sklearn)
- RandomForest (sklearn)
- NeuralNetworkClassifier (skorch)
- NeuralNetworkRegressor (skorch)
- XGBoostClassifier (xgboost)

Torch also has its own integration for custom models.

### Simple ML examples with scikit-learn.

A simple example which is very close to scikit-learn is as follows, for a logistic regression :

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression

# Create a synthetic dataset
N_EXAMPLE_TOTAL = 100
N_TEST = 20
x, y = make_classification(n_samples=N_EXAMPLE_TOTAL, class_sep=2, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=N_TEST / N_EXAMPLE_TOTAL, random_state=42
)

# Fix the quantization to 3 bits
model = LogisticRegression(n_bits=3)

# Fit the model
model.fit(X_train, y_train)

# We run prediction on non-encrypted data as a reference
y_pred_clear = model.predict(X_test, execute_in_fhe=False)

# We compile into an FHE model
model.compile(x)

# We then run the inference in FHE
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print("Comparison:", (y_pred_fhe == y_pred_clear))

# Output:
#   In clear  : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   In FHE    : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   Comparison: [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True]
```

We explain this in more detail in the documentation, and show how we have tried to mimic scikit-learn and torch APIs, to ease the adoption of **Concrete-ML** in [this page dedicated to scikit-learn](docs/user/howto/simple_example_sklearn.md) and in [this page dedicated to torch](docs/user/howto/simple_example_torch.md).

## For developers

### Project setup.

Installation steps are described in [project_setup.md](docs/dev/howto/project_setup.md).
Information about how to use Docker for development are available in [docker.md](docs/dev/howto/docker.md).

### Documenting.

Some information about how to build the documentation of **Concrete-ML** are [available](docs/dev/howto/documenting.md). Notably, our documentation is pushed to [https://docs.zama.ai/concrete-ml/](https://docs.zama.ai/concrete-ml/).

### Developing.

Some information about the infrastructure of **Concrete-ML** and some of the core elements we use are available [here](docs/dev/explanation/). Notably, an in-depth look at what is done in **Concrete-ML** is available in [onnx_use_for_compilation.md](docs/dev/explanation/onnx_use_for_compilation.md).

### Contributing.

Information about how to contribute is available in [contributing.md](docs/dev/howto/contributing.md).

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
