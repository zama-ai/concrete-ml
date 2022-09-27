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

Concrete-ML is a Privacy-Preserving Machine Learning (PPML) open-source set of tools built on top of [The Concrete Framework](https://github.com/zama-ai/concrete) by [Zama](https://github.com/zama-ai). It aims to simplify the use of fully homomorphic encryption (FHE) for data scientists to help them automatically turn machine learning models into their homomorphic equivalent. Particular care was given to the simplicity of the Python package in order to make it usable by any data scientist, even those without prior cryptography knowledge. Notably, the APIs are as close as possible to scikit-learn and torch APIs to simplify adoption by the users.

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

We explain this in more detail in the documentation, and show how we have tried to mimic scikit-learn and torch APIs, to ease the adoption of Concrete-ML. We refer the reader to [linear models](docs/built-in-models/linear.md), [tree-based models](docs/built-in-models/tree.md) and [neural networks](docs/built-in-models/neural-networks.md) documentations, which show how similar the APIs are to their non-FHE counterparts.

## Online demos and tutorials.

Various tutorials are proposed for the [built-in models](docs/built-in-models/ml_examples.md) and for [deep learning](docs/deep-learning/examples.md). In addition, we also list standalone use-cases:

- [MNIST](use_case_examples/mnist): a python and notebook showing a quantization-aware training (done with [Brevitas](https://github.com/Xilinx/brevitas) and following constraints of the package) and its corresponding use in Concrete-ML.
- [Encrypted sentiment analysis](use_case_examples/encrypted_sentiment_analysis): a gradio demo which predicts if a tweet / short message is positive, negative or neutral. Of course, in FHE! The corresponding application is directly available here(FIXME add link to https ://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis when it's online).

More generally, if you have built awesome projects using Concrete-ML, feel free to let us know and we'll link to it!

## Need support?

<a target="_blank" href="https://community.zama.ai">
  <img src="https://user-images.githubusercontent.com/5758427/191792238-b132e413-05f9-4fee-bee3-1371f3d81c28.png">
</a>

## License.

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
