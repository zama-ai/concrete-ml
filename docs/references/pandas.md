# Pandas

Concrete ML fully supports Pandas, allowing built-in models such as linear and tree-based models to use Pandas dataframes and series just as they would be used with NumPy arrays.

The table below summarizes current compatibility:

|         Methods          | Support Pandas dataframe |
| :----------------------: | :----------------------: |
|           fit            |            ✓             |
|         compile          |            ✓             |
| predict (fhe="simulate") |            ✓             |
| predict (fhe="execute")  |            ✓             |

## Example

The following example considers a `LogisticRegression` model on a simple classification problem.
A more advanced example can be found in the [Titanic use case notebook](../../use_case_examples/titanic/KaggleTitanic.ipynb), which considers a `XGBClassifier`.

```python
import numpy as np
import pandas as pd
from concrete.ml.sklearn import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create the data-set as a Pandas dataframe
X, y = make_classification(
    n_samples=250,
    n_features=30,
    n_redundant=0,
    random_state=2,
)
X, y = pd.DataFrame(X), pd.DataFrame(y)

# Retrieve train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the model
model = LogisticRegression(n_bits=8)

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model on the test set in clear
y_pred_clear = model.predict(X_test)

# Compile the model
model.compile(X_train)

# Perform the inference in FHE
y_pred_fhe = model.predict(X_test, fhe="execute")

# Assert that FHE predictions are the same as the clear predictions
print(
    f"{(y_pred_fhe == y_pred_clear).sum()} "
    f"examples over {len(y_pred_fhe)} have an FHE inference equal to the clear inference."
)

# Output:
    # 100 examples over 100 have an FHE inference equal to the clear inference.
```
