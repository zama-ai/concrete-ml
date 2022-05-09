# Use **Concrete ML** with scikit-learn

**Concrete-ML** is compatible with sklearn APIs such as Pipeline() or GridSearch(), which are popular model selection methods.

Here is a simple example of such a process:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from concrete.ml.sklearn.xgb import XGBClassifier


# Get dataset and split into train and test
X, y = load_breast_cancer(return_X_y=True)

# Split the train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Define our model
model = XGBClassifier(n_jobs=1, n_bits=3)

# Define the pipeline
# We will normalize the data and apply a PCA before fitting the model
pipeline = Pipeline([("standard_scaler", StandardScaler()), ("pca", PCA()), ("model", model)])

# Define the parameters to tune
param_grid = {
    "pca__n_components": [5, 10, 15],
    "model__max_depth": [2, 3, 5],
    "model__n_estimators": [5, 10, 20],
}

# Instantiate the grid search with 5-fold cross validation on all available cores
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy")

# Launch the grid search
grid.fit(X_train, y_train)

# Print the best parameters found
print(f"Best parameters found: {grid.best_params_}")

# Output:
#   Best parameters found:
#   {'model__max_depth': 5, 'model__n_bits': 6, 'model__n_estimators': 20, 'pca__n_components': 15}

# Currently we only focus on model inference in FHE
# The data transformation will be done in clear (client machine)
# while the model inference will be done in FHE on a server.
# The pipeline can be split into 2 parts:
#   1. data transformation
#   2. estimator
best_pipeline = grid.best_estimator_
data_transformation_pipeline = best_pipeline[:-1]
clf = best_pipeline[-1]

# Transform test set
X_train_transformed = data_transformation_pipeline.transform(X_train)
X_test_transformed = data_transformation_pipeline.transform(X_test)

# Evaluate the model on the test set (no FHE)
y_pred_clear = clf.predict(X_test_transformed)
print(f"Test accuracy: {(y_pred_clear == y_test).mean()}")

# Output:
#   Test accuracy: 0.9521

# Compile the model to FHE
clf.compile(X_train_transformed)

# Run the model in FHE
# Warning: this will take a while.
#          It is recommended to run this with a very small batch of example first
#          (e.g. N_TEST_FHE = 1)
# Note that here the encryption and decryption is done behind the scene.
N_TEST_FHE = 1
y_pred_fhe = clf.predict(X_test_transformed[:N_TEST_FHE], execute_in_fhe=True)

# Assert that FHE predictions are the same a the clear predictions
print(f"{(y_pred_fhe == y_pred_clear[:N_TEST_FHE]).sum()} "
      f"examples over {N_TEST_FHE} have a FHE inference equal to the clear inference.")
```

## Supported models

Currently, we support the following models in scikit-learn:

- LinearRegression
- LogisticRegression
- SVM (SVC and SVR)
- DecisionTreeClassifier
- PoissonRegressor
- GammaRegressor
- TweedieRegressor
- RandomForestClassifier
