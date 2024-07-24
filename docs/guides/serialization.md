# Serializing Built-In Models

This document explains how to serialize build-in models in Concrete ML.

## Introduction

Serialization allows you to dump a fitted and compiled model into a JSON string or file. You can then load the estimator back using the JSON object.

## Saving Models

All built-in models provide the following methods:

- `dumps`: Dumps the model as a string.
- `dump`: Dumps the model into a file.

For example, a logistic regression model can be dumped in a string as follows:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression

# Create the data for classification:
X, y = make_classification()

# Retrieve train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Instantiate, train and compile the model
model = LogisticRegression()
model.fit(X_train, y_train)
model.compile(X_train)

# Run the inference in FHE
y_pred_fhe = model.predict(X_test, fhe="execute")

# Dump the model in a string
dumped_model_str = model.dumps()

```

Similarly, it can be dumped into a file:

<!--pytest-codeblocks:cont-->

```python
from pathlib import Path

dumped_model_path = Path("logistic_regression_model.json")

# Any kind of file-like object can be used 
with dumped_model_path.open("w") as f:

    # Dump the model in a file
    model.dump(f)
```

Alternatively, Concrete ML provides two equivalent global functions:

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.common.serialization.dumpers import dump, dumps

# Dump the model in a string
dumped_model_str = dumps(model)

# Any kind of file-like object can be used 
with dumped_model_path.open("w") as f:

    # Dump the model in a file
    dump(model, f)
```

{% hint style="warning" %}
Some parameters used for instantiating Quantized Neural Network models are not supported for serialization. In particular, you cannot serialize a model that was instantiated using callable objects for the `train_split` and `predict_nonlinearity` parameters or with `callbacks` being enabled.
{% endhint %}

## Loading Models

You can load a built-in model using the following functions:

- `loads`: Loads the model from a string.
- `load`: Loads the model from a file.

{% hint style="warning" %}
A loaded model must be compiled once again to execute the inference in
FHE or with simulation because the underlying FHE circuit is currently not serialized.
This recompilation is not required when FHE mode is disabled.
{% endhint %}

The same logistic regression model can be loaded as follows:

<!--pytest-codeblocks:cont-->

```python
import numpy
from concrete.ml.common.serialization.loaders import load, loads

# Load the model from a string
loaded_model = loads(dumped_model_str)

# Any kind of file-like object can be used 
with dumped_model_path.open("r") as f:

    # Load the model from a file
    loaded_model = load(f)

# Compile the model
loaded_model.compile(X_train)

# Run the inference in FHE using the loaded model
y_pred_fhe_loaded = loaded_model.predict(X_test, fhe="execute")

print("Predictions are equal:", numpy.array_equal(y_pred_fhe, y_pred_fhe_loaded))

# Output:
#   Predictions are equal: True
```
