# Encrypted training

Concrete ML offers the possibility to train [SGD Logistic Regression](../references/api/concrete.ml.sklearn.linear_model.md#class-sgdclassifier) on encrypted data. The [logistic regression training](../advanced_examples/LogisticRegressionTraining.ipynb) example shows this feature in action.

This example shows how to instantiate a logistic regression model that trains on encrypted data:

```python
from concrete.ml.sklearn import SGDClassifier
parameters_range = (-1.0, 1.0)

model = SGDClassifier(
    random_state=42,
    max_iter=50,
    fit_encrypted=True,
    parameters_range=parameters_range,
)
```

To activate encrypted training simply set `fit_encrypted=True` in the constructor. If this value is not set, training is performed on clear data using `scikit-learn` gradient descent.

Next, to perform the training on encrypted data, call the `fit` function with the `fhe="execute"` argument:

<!--pytest-codeblocks:skip-->

```python
model.fit(X_binary, y_binary, fhe="execute")
```

{% hint style="info" %}
Training on encrypted data provides the highest level of privacy but is slower than training on clear data. Federated learning is an alternative approach, where data privacy can be ensured by using a trusted gradient aggregator, coupled with optional _differential privacy_ instead of encryption. Concrete ML can import linear models, including logistic regression, that are trained using federated learning using the [`from_sklearn` function](linear.md#pre-trained-models).
{% endhint %}

## Training configuration

The `max_iter` parameter controls the number of batches that are processed by the training algorithm.

The `parameters_range` parameter determines the initialization of the coefficients and the bias of the logistic regression. It is recommended to give values that are close to the min/max of the training data. It is also possible to normalize the training data so that it lies in the range $$[-1, 1]$$.

## Capabilities and Limitations

The logistic model that can be trained uses Stochastic Gradient Descent (SGD) and quantizes for data, weights, gradients and the error measure. It currently supports training 6-bit models, training both the coefficients and the bias.

The `SGDClassifier` does not currently support training models with other values for the bit-widths. The execution time to train a model is proportional to the number of features and the number of training examples in the batch. The `SGDClassifier` training does not currently support client/server deployment for training.
