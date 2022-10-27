<p align="center">
<!-- product name logo -->
  <img width=600 src="https://user-images.githubusercontent.com/5758427/188829741-8503b6c3-98ca-4285-9955-455508f50863.png">
</p>
<p align="center">
<!-- Version badge using shields.io -->
  <a href="https://github.com/zama-ai/concrete-ml/releases">
    <img src="https://img.shields.io/github/v/release/zama-ai/concrete-ml?style=flat-square">
  </a>
<!-- Link to docs badge using shields.io -->
  <a href="https://docs.zama.ai/concrete-ml">
    <img src="https://img.shields.io/badge/read-documentation-yellow?style=flat-square">
  </a>
<!-- Link to tutorials badge using shields.io -->
  <a href="https://github.com/zama-ai/concrete-ml/tree/release/0.3.x/docs/advanced_examples">
    <img src="https://img.shields.io/badge/tutorials-and%20demos-orange?style=flat-square">
  </a>
<!-- Community forum badge using shields.io -->
  <a href="https://community.zama.ai/c/concrete-ml">
    <img src="https://img.shields.io/badge/community%20forum-online-brightgreen?style=flat-square">
  </a>
<!-- Open source badge using shields.io -->
  <a href="https://docs.zama.ai/concrete-ml/developer-guide/contributing">
    <img src="https://img.shields.io/badge/we're%20open%20source-contributing.md-blue?style=flat-square">
  </a>
<!-- Follow on twitter badge using shields.io -->
  <a href="https://twitter.com/zama_fhe">
    <img src="https://img.shields.io/twitter/follow/zama_fhe?color=blue&style=flat-square">
  </a>
</p>

Concrete-ML is a Privacy-Preserving Machine Learning (PPML) open-source set of tools built on top of [The Concrete Framework](https://github.com/zama-ai/concrete) by [Zama](https://github.com/zama-ai). It aims to simplify the use of fully homomorphic encryption (FHE) for data scientists to help them automatically turn machine learning models into their homomorphic equivalent. Concrete-ML was designed with ease-of-use in mind, so that data scientists can use it without knowledge of cryptography. Notably, the Concrete-ML model classes are similar to those in scikit-learn and it is also possible to convert PyTorch models to FHE.

## Main features.

Data scientists can use models with APIs which are close to the frameworks they use, with additional options to run inferences in FHE.

Concrete-ML features:

- built-in models, which are ready-to-use FHE-friendly models with a user interface that is equivalent to their the scikit-learn and XGBoost counterparts
- support for customs models that can use quantization aware training. These are developed by the user using pytorch or keras/tensorflow and are imported into Concrete-ML through ONNX

## Installation.

Depending on your OS, Concrete-ML may be installed with Docker or with pip:

|               OS / HW                | Available on Docker | Available on pip |
| :----------------------------------: | :-----------------: | :--------------: |
|                Linux                 |         Yes         |       Yes        |
|               Windows                |         Yes         |   Coming soon    |
|     Windows Subsystem for Linux      |         Yes         |       Yes        |
|            macOS (Intel)             |         Yes         |       Yes        |
| macOS (Apple Silicon, ie M1, M2 etc) |         Yes         |   Coming soon    |

Note: Concrete-ML only supports Python `3.8` and `3.9`. Platforms like [Kaggle](https://www.kaggle.com) or [Google Colab](https://colab.research.google.com) use Python `3.7` which is a deprecated version and is not currently supported in Concrete-ML.

### Docker.

To install with Docker, pull the `concrete-ml` image as follows:

`docker pull zamafhe/concrete-ml:latest`

### Pip.

To install Concrete-ML from PyPi, run the following:

```
pip install -U pip wheel setuptools
pip install concrete-ml
```

You can find more detailed installation instructions in [this part of the documentation](docs/getting-started/pip_installing.md)

## A simple Concrete-ML example with scikit-learn.

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

# Compile into a FHE model
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

This example is explained in more detail in the [linear model documentation](docs/built-in-models/linear.md). Concrete-ML built-in models
have APIs that are almost identical to their scikit-learn counterparts. It is also possible to convert PyTorch networks to FHE with the Concrete-ML conversion APIs. Please refer to the [linear models](docs/built-in-models/linear.md), [tree-based models](docs/built-in-models/tree.md) and [neural networks](docs/built-in-models/neural-networks.md) documentation for more examples, showing the scikit-learn-like API of the built-in
models.

## Documentation.

Full, comprehensive documentation is available here: [https://docs.zama.ai/concrete-ml](https://docs.zama.ai/concrete-ml).

## Online demos and tutorials.

Various tutorials are proposed for the [built-in models](docs/built-in-models/ml_examples.md) and for [deep learning](docs/deep-learning/examples.md). In addition, several complete use-cases are explored:

- [MNIST](use_case_examples/mnist):a python script and notebook showing quantization-aware training following FHE constraints. The model is implemented with [Brevitas](https://github.com/Xilinx/brevitas) and is converted to FHE with Concrete-ML.

- [Titanic](use_case_examples/titanic/KaggleTitanic.ipynb): a notebook, which gives a solution to the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/). Done with XGBoost from Concrete-ML. It comes as a companion of [Kaggle notebook](https://www.kaggle.com/code/concretemlteam/titanic-with-privacy-preserving-machine-learning), and was the subject of a blogpost in [KDnuggets](https://www.kdnuggets.com/2022/08/machine-learning-encrypted-data.html).

More generally, if you have built awesome projects using Concrete-ML, feel free to let us know and we'll link to it!

## Need support?

<a target="_blank" href="https://community.zama.ai">
  <img src="https://user-images.githubusercontent.com/5758427/191792238-b132e413-05f9-4fee-bee3-1371f3d81c28.png">
</a>

## License.

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
