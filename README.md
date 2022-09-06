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

- [documentation](https://docs.zama.ai/concrete-ml)
- [community website](https://community.zama.ai/c/concrete-ml)
- [demos](https://docs.zama.ai/concrete-ml/built-in-models/ml_examples)

## For end users

### Installation.

The preferred way to use **Concrete-ML** is through Docker. You can get our Docker image by pulling the latest Docker image:

`docker pull zamafhe/concrete-ml:latest`

To install **Concrete-ML** from PyPi, run the following:

`pip install concrete-ml`

You can find more detailed installation instructions in [pip_installing.md](docs/getting-started/pip_installing.md)

### Supported models.

Here is a list of ML algorithms currently supported in this library:

- LinearRegression (sklearn)
- Lasso, Ridge, ElasticNet (sklearn)
- LogisticRegression (sklearn)
- SVM - LinearSVC and LinearSVR (sklearn)
- DecisionTreeClassifier (sklearn)
- DecisionTreeRegressor (sklearn)
- RandomForestClassifier (sklearn)
- NeuralNetworkClassifier (skorch)
- NeuralNetworkRegressor (skorch)
- XGBoostClassifier (xgboost)
- XGBoostRegressor (xgboost)

Torch also has its own integration for custom models.

### Simple ML examples with scikit-learn.

A simple example which is very close to scikit-learn is as follows, for a logistic regression :

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression

# Create the data for classification
x, y = make_classification(n_samples=100, class_sep=2, n_features=4, random_state=42)

# Retrieve train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=10, random_state=42
)

# Fix the number of bits to used for quantization 
model = LogisticRegression(n_bits=2)

# Fit the model
model.fit(X_train, y_train)

# Run the predictions on non-encrypted data as a reference
y_pred_clear = model.predict(X_test, execute_in_fhe=False)

# Compile into an FHE model
model.compile(x)

# Run the inference in FHE
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print(f"Comparison: {int((y_pred_fhe == y_pred_clear).sum()/len(y_pred_fhe)*100)}% similar")

# Output:
#  In clear  : [0 0 0 1 0 1 0 1 1 1]
#  In FHE    : [0 0 0 1 0 1 0 1 1 1]
#  Comparison: 100% similar
```

We explain this in more detail in the documentation, and show how we have tried to mimic scikit-learn and torch APIs, to ease the adoption of **Concrete-ML**. We refer the reader to [linear models](docs/built-in-models/linear.md), [tree-based models](docs/built-in-models/tree.md) and [neural networks](docs/built-in-models/neural-networks.md) documentations, which show how similar our APIs are to their non-FHE counterparts.

## For developers

### Project setup.

Installation steps are described in [project_setup.md](docs/developer-guide/project_setup.md).
Information about how to use Docker for development are available in [docker_setup.md](docs/developer-guide/docker_setup.md).

### Documenting.

Some information about how to build the documentation of **Concrete-ML** are [available](docs/developer-guide/documenting.md). Notably, our documentation is pushed to [https://docs.zama.ai/concrete-ml/](https://docs.zama.ai/concrete-ml/).

### Developing.

### Contributing.

Information about how to contribute is available in [contributing.md](docs/developer-guide/contributing.md).

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
