# Serializing Built-In Models

Concrete ML has support for serializing all available built-in models. Using this feature, one can
dump a fitted and compiled model into a JSON string or file. The estimator can then be loaded back
using the JSON object.

## Saving Models

All built-in models provide the following methods:

- `dumps`: dumps the model as a string.
- `dump`: dumps the model into a file.

For example, a logistic regression model can be dumped in a string as below.

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

Similarly, it can be dumped into a file.

<!--pytest-codeblocks:cont-->

```python
from pathlib import Path

dumped_model_path = Path("logistic_regression_model.json")

# Any kind of file-like object can be used 
with dumped_model_path.open("w") as f:

    # Dump the model in a file
    model.dump(f)
```

Alternatively, Concrete ML provides two equivalent global functions.

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
Some parameters used for instantiating Quantized Neural Network models are not supported for
serialization. In particular, one cannot serialize a model that was instantiated using callable
objects for the `train_split` and `predict_nonlinearity` parameters or with `callbacks` being
enabled.
{% endhint %}

## Loading Models

Loading a built-in model is possible through the following functions:

- `loads`: loads the model from a string.
- `load`: loads the model from a file.

{% hint style="warning" %}
A loaded model is required to be compiled once again in order for a user to be able to execute the inference in
FHE or with simulation. This is because the underlying FHE circuit is currently not serialized.
There is not required when FHE mode is disabled.
{% endhint %}

The above logistic regression model can therefore be loaded as below.

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
