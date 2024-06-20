# Linear models

This page explains Concrete ML linear models for both classification and regression. These models are based on [scikit-learn](https://scikit-learn.org/stable/) linear models.

## Supported models for encrypted inference

The following models are supported for training on clear data and predicting on encrypted data. Their API is similar the one of [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model). These models are also compatible with some of scikit-learn's main workflows, such as `Pipeline()` and `GridSearch()`.

|                                             Concrete ML                                              |                                                                         scikit-learn                                                                         |
| :--------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   [LinearRegression](../references/api/concrete.ml.sklearn.linear_model.md#class-linearregression)   |    [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)    |
| [LogisticRegression](../references/api/concrete.ml.sklearn.linear_model.md#class-logisticregression) | [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) |
|              [LinearSVC](../references/api/concrete.ml.sklearn.svm.md#class-linearsvc)               |                       [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)                        |
|              [LinearSVR](../references/api/concrete.ml.sklearn.svm.md#class-linearsvr)               |                       [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)                        |
|       [PoissonRegressor](../references/api/concrete.ml.sklearn.glm.md#class-poissonregressor)        |    [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor)    |
|       [TweedieRegressor](../references/api/concrete.ml.sklearn.glm.md#class-tweedieregressor)        |    [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor)    |
|         [GammaRegressor](../references/api/concrete.ml.sklearn.glm.md#class-gammaregressor)          |       [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor)       |
|              [Lasso](../references/api/concrete.ml.sklearn.linear_model.md#class-lasso)              |                    [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)                     |
|              [Ridge](../references/api/concrete.ml.sklearn.linear_model.md#class-ridge)              |                    [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)                     |
|         [ElasticNet](../references/api/concrete.ml.sklearn.linear_model.md#class-elasticnet)         |             [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)             |
|       [SGDRegressor](../references/api/concrete.ml.sklearn.linear_model.md#class-sgdregressor)       |                           [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)                           |

## Supported models for encrypted training

In addition to predicting on encrypted data, the following models  support training on encrypted data.

|       [SGDClassifier](../references/api/concrete.ml.sklearn.linear_model.md#class-sgdclassifier)       |                           [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)                           |

## Quantization parameters

The `n_bits` parameter controls the bit-width of the inputs and weights of the linear models. Linear models do not use table lookups and thus alllows weight and inputs to be high precision integers.

For models with input dimensions up to `300`, the parameter `n_bits` can be set to `8` or more. When the input dimensions are larger, `n_bits` must be reduced to `6-7`. In many cases, quantized models can preserve all performance metrics compared to the non-quantized float models from scikit-learn when `n_bits` is down to `6`. You should validate accuracy on held-out test sets and adjust `n_bits` accordingly.

{% hint style="warning" %}

For optimal results, you can use standard or min-max normalization to achieve a similar distribution of individual features. When there are many one-hot features, consider [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  as a pre-processing stage.

For a more detailed comparison of the impact of such pre-processing,  please refer to [the logistic regression notebook](../advanced_examples/LogisticRegression.ipynb).

{% endhint %}

## Pre-trained models

You can convert an already trained scikit-learn linear model to a Concrete ML one by using the [`from_sklearn_model`](../references/api/concrete.ml.sklearn.base.md#classmethod-from_sklearn_model) method. See [the following example](linear.md#loading-a-pre-trained-model).

## Example

The following example shows how to train a LogisticRegression model on a simple data-set and then use FHE to perform inference on encrypted data. You can find a more complete example in the [LogisticRegression notebook](../tutorials/ml_examples.md).

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LogisticRegression

# Create the data for classification:
X, y = make_classification(
    n_features=30,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
    n_samples=250,
)

# Retrieve train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the model:
model = LogisticRegression(n_bits=8)

# Fit the model:
model.fit(X_train, y_train)

# Evaluate the model on the test set in clear:
y_pred_clear = model.predict(X_test)

# Compile the model:
model.compile(X_train)

# Perform the inference in FHE:
y_pred_fhe = model.predict(X_test, fhe="execute")

# Assert that FHE predictions are the same as the clear predictions:
print(
    f"{(y_pred_fhe == y_pred_clear).sum()} examples over {len(y_pred_fhe)} "
    "have an FHE inference equal to the clear inference."
)

# Output:
#  100 examples over 100 have an FHE inference equal to the clear inference
```

## Model accuracy

The figure below compares the decision boundary of the FHE classifier and a scikit-learn model executed in clear. You can find the complete code in the [LogisticRegression notebook](../tutorials/ml_examples.md).

The overall accuracy scores are identical (93%) between the scikit-learn model (executed in the clear) and the Concrete ML one (executed in FHE). In fact, quantization has little impact on the decision boundaries, as linear models can use large precision numbers when quantizing inputs and weights in Concrete ML. Additionally, as the linear models do not use [Programmable Boostrapping](../getting-started/concepts.md#cryptography-concepts), the FHE computations are always exact, irrespective of the [PBS error tolerance setting](../explanations/advanced_features.md#approximate-computations). This ensures that the FHE predictions are always identical to the quantized clear ones.

![Sklearn model decision boundaries](../figures/logistic_regression_clear.png) ![FHE model decision boundaries](../figures/logistic_regression_fhe.png)

## Loading a pre-trained model

An alternative to the example above is to train a scikit-learn model in a separate step and then to convert it to Concrete ML.

```
from sklearn.linear_model import LogisticRegression as SKlearnLogisticRegression

# Instantiate the model:
model = SKlearnLogisticRegression()

# Fit the model:
model.fit(X_train, y_train)

cml_model = LogisticRegression.from_sklearn_model(model, X_train, n_bits=8)

# Compile the model:
cml_model.compile(X_train)

# Perform the inference in FHE:
y_pred_fhe = cml_model.predict(X_test, fhe="execute")


```
