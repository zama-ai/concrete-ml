# Pandas

Concrete-ML provides partial support for Pandas, with most available models (linear and tree-based models) usable on Pandas dataframes the same way they would be used with Numpy arrays.

The table below summarizes the current compatibility:

|            Methods             | Support Pandas dataframe |
| :----------------------------: | :----------------------: |
|              fit               |            ✓             |
|            compile             |            ✗             |
| predict (execute_in_fhe=False) |            ✓             |
| predict (execute_in_fhe=True)  |            ✓             |

## Example

The following example uses a `LogisticRegression` model on a simple classification problem. A more advanced example can be found in the [KaggleTitanic notebook](ml_examples.md).

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
model = LogisticRegression(n_bits=2)

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model on the test set in clear
y_pred_clear = model.predict(X_test)

# Compile the model
model.compile(X_train.to_numpy())

# Perform the inference in FHE
# Warning: this will take a while. It is recommended to run this with a very small batch of
# example first (e.g. N_TEST_FHE = 1)
# Note that here the encryption and decryption is done behind the scene.
N_TEST_FHE = 1
y_pred_fhe = model.predict(X_test.head(N_TEST_FHE), execute_in_fhe=True)

# Assert that FHE predictions are the same as the clear predictions
print(f"{(y_pred_fhe == y_pred_clear[:N_TEST_FHE]).sum()} "
      f"examples over {N_TEST_FHE} have a FHE inference equal to the clear inference.")

# Output:
#  1 examples over 1 have a FHE inference equal to the clear inference
```

















































































































































































































































