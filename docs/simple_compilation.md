# Converting a simple model to FHE

A simple example which is very close to scikit-learn is as follows, for a logistic regression.

First, we import classical ML packages.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

Second, we import `LogisticRegression`, but not the one from scikit-learn, the one from **Concrete-ML**, which shares the same API.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.sklearn import LogisticRegression
```

Third, we create a synthetic dataset.

<!--pytest-codeblocks:cont-->

```python
N_EXAMPLE_TOTAL = 100
N_TEST = 20
x, y = make_classification(n_samples=N_EXAMPLE_TOTAL,
    class_sep=2, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=N_TEST / N_EXAMPLE_TOTAL, random_state=42
)
```

Fourth, we need to define the quantization to 3 bits, and can then fit the model.

<!--pytest-codeblocks:cont-->

```python
model = LogisticRegression(n_bits=3)
model.fit(X_train, y_train)
```

Fifth, we can run the predictions in clear (as a reference), i.e., without Fully Homomorphic Encryption.

<!--pytest-codeblocks:cont-->

```python
y_pred_clear = model.predict(X_test, execute_in_fhe=False)
```

Sixth, we compile the model, to get its equivalent FHE counterpart.

<!--pytest-codeblocks:cont-->

```python
model.compile(x)
```

Seventh, we can then run inferences, but now, in FHE.

<!--pytest-codeblocks:cont-->

```python
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)
```

If you run this, you should see that the inferences in clear and in FHE are equal.

<!--pytest-codeblocks:cont-->

```python
print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print("Comparison:", (y_pred_fhe == y_pred_clear))

# Output:
#   In clear  : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   In FHE    : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   Comparison: [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True]
```
