# Linear Models

Concrete-ML provides several of the most popular linear models for `regression` or `classification` that can be found in [Scikit-learn](https://scikit-learn.org/stable/):

|                                                Concrete-ML                                                |                                                                         scikit-learn                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   [LinearRegression](../developer-guide/api/concrete.ml.sklearn.linear_model.md#class-linearregression)   |    [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)    |
| [LogisticRegression](../developer-guide/api/concrete.ml.sklearn.linear_model.md#class-logisticregression) | [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) |
|              [LinearSVC](../developer-guide/api/concrete.ml.sklearn.svm.md#class-linearsvc)               |                       [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)                        |
|              [LinearSVR](../developer-guide/api/concrete.ml.sklearn.svm.md#class-linearsvr)               |                       [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)                        |
|       [PoissonRegressor](../developer-guide/api/concrete.ml.sklearn.glm.md#class-poissonregressor)        |    [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor)    |
|       [TweedieRegressor](../developer-guide/api/concrete.ml.sklearn.glm.md#class-tweedieregressor)        |    [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor)    |
|         [GammaRegressor](../developer-guide/api/concrete.ml.sklearn.glm.md#class-gammaregressor)          |       [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor)       |
|              [Lasso](../developer-guide/api/concrete.ml.sklearn.linear_model.md#class-lasso)              |                    [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)                     |
|              [Ridge](../developer-guide/api/concrete.ml.sklearn.linear_model.md#class-ridge)              |                    [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)                     |
|         [ElasticNet](../developer-guide/api/concrete.ml.sklearn.linear_model.md#class-elasticnet)         |             [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)             |

Using these models in FHE is extremely similar to what can be done with scikit-learn's [API](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model), making it easy for data scientists that are used to this framework to get started with Concrete ML.

Models are also compatible with some of scikit-learn's main worflows, such as `Pipeline()` or `GridSearch()`.

## Example

Here's an example of how to use this model in FHE on a simple dataset below. A more complete example can be found in the [LogisticRegression notebook](ml_examples.md).

```python
import numpy
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression

# Create the data for classification
X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
    n_samples=100,
)

# Retrieve train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the model
model = LogisticRegression(n_bits=2)

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model on the test set in clear
y_pred_clear = model.predict(X_test)

# Compile the model
model.compile(X_train)

# Perform the inference in FHE
# Note that here the encryption and decryption is done behind the scene.
# It is recommended to run this with a very small batch of
# examples first (e.g. N_TEST_FHE = 3)
N_TEST_FHE = 3
y_pred_fhe = numpy.array([
  model.predict([sample], execute_in_fhe=True)[0]
  for sample in tqdm(X_test[:N_TEST_FHE])
])

# Assert that FHE predictions are the same as the clear predictions
print(f"{(y_pred_fhe == y_pred_clear[:N_TEST_FHE]).sum()} "
      f"examples over {N_TEST_FHE} have a FHE inference equal to the clear inference.")

# Output:
#  3 examples over 3 have a FHE inference equal to the clear inference
```

We can then plot the decision boundary of the classifier and then compare those results with a scikit-learn model executed in clear. The complete code can be found in the [LogisticRegression notebook](ml_examples.md).

![Plaintext model decision boundaries](../figures/logistic_regression_clear.png) ![FHE model decision boundarires](../figures/logistic_regression_fhe.png)

We can clearly observe the impact of quantization over the decision boundaries in the FHE model, breaking the initial lines into broken lines with steps. However, this does not change the overall score as both models output the same accuracy (90%).

In fact, the quantization process may sometimes create some artifacts that could lead to a decrease in performance. Still, the impact of those artifacts is often minor when considering linear models, making FHE models reach similar scores as their equivalent clear ones.
