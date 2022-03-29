# Concrete ML

Concrete ML is an open-source set of tools which aims to simplify the use of fully homomorphic encryption (FHE) for data scientists. Particular care was given to the simplicity of our python package, in order to make it usable by any data scientist, without any prior cryptography knowledge. Notably, our APIs are as close as possible from scikit-learn and torch APIs, to simplify adoption by our users.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Links](#links)
- [For end users](#for-end-users)
  - [Installation](#installation)
  - [Simple ML examples with scikit-learn and torch comparison](#simple-ml-examples-with-scikit-learn-and-torch-comparison)
- [For developers](#for-developers)
  - [Project setup](#project-setup)
  - [Documenting](#documenting)
  - [Developing](#developing)
  - [Contributing](#contributing)
- [License](#license)

<!-- mdformat-toc end -->

## Links

- [documentation](https://docs.zama.ai/concrete-ml/main/)
- [community website](https://community.zama.ai/c/concrete-ml)
- [demos](https://docs.zama.ai/concrete-ml/main/user/advanced_examples/index.html)

## For end users

### Installation

The preferred way to use Concrete ML is through docker. You can get our docker image by pulling the latest docker image:

`docker pull zamafhe/concrete-ml:latest`

To install Concrete ML from PyPi, run the following:

`pip install concrete-ml`

You can find more detailed installation instructions in [installing.md](docs/user/basics/installing.md)

### Simple ML examples with scikit-learn and torch comparison

A simple example which is very close to scikit-learn is as follows, for a linear regression:

```python
from sklearn.datasets import make_classification
from concrete.ml.sklearn import LinearRegression

# Create a synthetic dataset
x, y = make_classification(n_samples=200, class_sep=2, n_features=4, random_state=42)

# Fix the quantization to 2 bits
model = LinearRegression(n_bits=2)

# Fit the model
model, _ = model.fit_benchmark(x, y)

nb_inputs = 10

# We run prediction on non-encrypted data as a reference
y_pred = model.predict(x[:nb_inputs], execute_in_fhe=False)

# We compile into an FHE model
model.compile(x)

# We then run the inference in FHE
y_pred_fhe = model.predict(x[:nb_inputs], execute_in_fhe=True)

print("In clear  :", y_pred.flatten())
print("In FHE    :", y_pred_fhe.flatten())
print("Comparison:", (y_pred_fhe == y_pred).flatten())

# Will print
# In clear  : [1.17241192 1.17241192 0.         1.17241192 0.58620596 1.17241192
#  1.17241192 0.58620596 0.         0.58620596]
# In FHE    : [1.17241192 1.17241192 0.         1.17241192 0.58620596 1.17241192
#  1.17241192 0.58620596 0.         0.58620596]
# Comparison: [ True  True  True  True  True  True  True  True  True  True]
```

We explain this into more details in the documentation, and show how we have tried to mimic scikit-learn and torch APIs, to ease the adoption of **Concrete ML** in [this page dedicated to scikit-learn](docs/howto/simple_example_sklearn.md) and in [this page dedicated to torch](docs/howto/simple_example_torch.md).

## For developers

### Project setup

Installation steps are described in [project_setup.md](docs/dev/howto/project_setup.md).
Information about how to use Docker for development are available in [docker.md](docs/dev/howto/docker.md).

### Documenting

Some information about how to build the documentation of Concrete ML are [available](docs/dev/howto/documenting.md). Notably, our documentation is pushed to [https://docs.zama.ai/concrete-ml/](https://docs.zama.ai/concrete-ml/).

### Developing

Some information about the infrastructure of Concrete ML and some of the core elements we use are available [here](docs/dev/explanation/). Notably, an in-depth look at what is done in Concrete ML is available in [onnx_use_for_compilation.md](docs/dev/explanation/onnx_use_for_compilation.md).

### Contributing

Information about how to contribute are available in [contributing.md](docs/dev/howto/contributing.md).

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
