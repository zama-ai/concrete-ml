# Tree Models

## Scikit-learn

**Concrete-ML** provides several of the most popular tree models `classification` that can be found in [scikit-learn](https://scikit-learn.org/stable/):

|                                                                 Concrete-ML                                                                 |                                                                           scikit-learn                                                                           |
| :-----------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DecisionTreeClassifier](_apidoc/concrete.ml.sklearn.html?highlight=decisiontreeclassifier#concrete.ml.sklearn.tree.DecisionTreeClassifier) |     [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)     |
|  [RandomForestClassifier](_apidoc/concrete.ml.sklearn.html?highlight=randomforestclassifier#concrete.ml.sklearn.rf.RandomForestClassifier)  | [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) |

Using those models in FHE is extremely similar to what can be done with scikit-learn's  \[API\](https://scikit-learn.org/stable/modules/classes. Any data scientists that are used to this framework should find the FHE tools very straightforward. More details about compiling and running any simple models can be found [here](simple_compilation.md).

Models from **Concrete-ML** are also compatible with some of scikit-learn's main worflows, such as `Pipeline()` or `GridSearch()`. See below for an example on how to use both.

## XGBoost

In addition to our support for scikit-learn, we also provide [XGBoost](https://xgboost.ai/) 's  `XGBClassifier`:

|                                       Concrete-ML                                       |                                                XGboost                                                 |
| :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| [XGBClassifier](_apidoc/concrete.ml.sklearn.html#concrete.ml.sklearn.xgb.XGBClassifier) | [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) |

## Training and predicting with Concrete-ML

All of the **training process is handled by scikit-learn or XGBoost**. Therefore, any users should refer to the associated framework's documentation considering details and parameters about the training part. **Concrete-ML** enables executing the trained model's inferences on encrypted data using FHE.

## Example

Here's an example of how to use this model in FHE on a popular dataset using some of scikit-learn's most popular preprocessing tools.
A more complete example can be found in the [XGBClassifier notebook](ml_examples.md).

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
    "pca__n_components": [2, 5, 10, 15],
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
#  Best parameters found: {'model__max_depth': 5, 'model__n_estimators': 10, 'pca__n_components': 5}

# Currently we only focus on model inference in FHE
# The data transformation will be done in clear (client machine)
# while the model inference will be done in FHE on a server.
# The pipeline can be split into 2 parts:
#   1. data transformation
#   2. estimator
best_pipeline = grid.best_estimator_
data_transformation_pipeline = best_pipeline[:-1]
model = best_pipeline[-1]

# Transform test set
X_train_transformed = data_transformation_pipeline.transform(X_train)
X_test_transformed = data_transformation_pipeline.transform(X_test)

# Evaluate the model on the test set in clear
y_pred_clear = model.predict(X_test_transformed)
print(f"Test accuracy in clear: {(y_pred_clear == y_test).mean():0.2f}")

# Output:
#  Test accuracy: 0.98

# Compile the model to FHE
model.compile(X_train_transformed)

# Perform the inference in FHE
# Warning: this will take a while. It is recommended to run this with a very small batch of
# example first (e.g. N_TEST_FHE = 1)
# Note that here the encryption and decryption is done behind the scene.
N_TEST_FHE = 1
y_pred_fhe = model.predict(X_test_transformed[:N_TEST_FHE], execute_in_fhe=True)

# Assert that FHE predictions are the same as the clear predictions
print(f"{(y_pred_fhe == y_pred_clear[:N_TEST_FHE]).sum()} "
      f"examples over {N_TEST_FHE} have a FHE inference equal to the clear inference.")

# Output:
#  1 examples over 1 have a FHE inference equal to the clear inference
```

## Visual comparison

Using the above example, we can then plot how the model classifies the inputs and then compare those results with the XGBoost model executed in clear. A 6 bits model is also given in order to better understand the impact of quantization on classification.
Similar plots can be found in the [Classifier Comparison notebook](ml_examples.md).

Let's plot the decision boundaries of both model.

### Classification Decision Boundaries

| ![XGBClassifier comparison](figures/xgb_comparison_pipeline.png) |
| :--------------------------------------------------------------: |
|                *XGBClassifier models comparison*                 |
|                                                                  |

We can clearly observe the impact of quantization over the decision boundaries in the FHE models, especially with the 3 bits model, where only three main decision boundaries can be observed. This results in a small decrease of accuracy of about 7% compared to the initial XGBoost classifier. Besides, using 6 bits of quantization makes the model reach 93% of accuracy, drastically reducing this difference to only 1.7%.

In fact, the quantization process may sometimes create some artifacts that could lead to a decrease in performance. Still, the impact of those artifacts is often minor when considering small tree-based models, making FHE models reach similar scores as their equivalent clear ones.
