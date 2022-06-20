# Pandas

## Pandas support

**Concrete-ML** provides some support for one of the most popular Python library for data analysis, Pandas.
This means that most available models (linear and tree-based models) can be used on pandas dataframe the same way any data scientists would use them with numpy arrays.

More specifically, for any models (except Quantized Neural Networks) :

|            Methods             | Support Pandas dataframe |
| :----------------------------: | :----------------------: |
|              fit               |            ✓             |
|            compile             |            ✗             |
| predict (execute_in_fhe=False) |            ✓             |
| predict (execute_in_fhe=True)  |            ✓             |

## Example

Here's an example using the `LogisticRegression` model on a simple classification problem. A more advanced example can be found in the [KaggleTitanic notebook](advanced_examples.md).

```python
import numpy as np
import pandas as pd
from concrete.ml.sklearn import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create the data set as a Pandas dataframe
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    random_state=2,
)
X, y = pd.DataFrame(X), pd.DataFrame(y)

# Retrieve train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the model
n_bits = {"net_inputs": 5, "op_inputs": 5, "op_weights": 2, "net_outputs": 8}
model = LogisticRegression(n_bits=n_bits)

# Fit the model
model.fit(X_train, y_train)

# Compile the model
model.compile(X_train.to_numpy())

# Perform the inference in FHE on the first data point
y_pred = model.predict(X_test.head(1), execute_in_fhe=True)

# Check if the prediction is correct
print(np.array_equal(y_pred, y_test.head(1).to_numpy()[0]))

# Output:
#   True

```

## Pre-processing with Pandas

Currently, it is not possible to execute most pre-processing tools from Pandas in FHE, such as dropping columns, adding features or common methods like `fillna` and `isnull`. It has to be done previously to any FHE call.
