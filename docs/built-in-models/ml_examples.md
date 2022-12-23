# Built-in Model Examples

These examples illustrate the basic usage of built-in Concrete-ML models. For more examples showing how to train high-accuracy models on more complex data-sets, see the [Demos and Tutorials](../getting-started/showcase.md) section.

## FHE constraints considerations

In Concrete-ML, built-in linear models are exact equivalents to their scikit-learn counterparts. Indeed, since they do not apply any non-linearity during inference, these models are very fast (~1ms FHE inference time) and can use high precision integers, between 20-25bits.

Tree-based models apply non-linear functions that enable comparisons of inputs and the trained thresholds. Thus, they are limited with respect to the number of bits used to represent the inputs, but, as these examples show, in practice 5-6 bits are sufficient to exactly reproduce the behavior of their scikit-learn counter-part models.

As shown in the examples below, built-in neural networks can be configured to work with user-specified accumulator sizes, which allows the user to adjust the speed/accuracy tradeoff.

{% hint style="info" %}
It is recommended to use [simulation](../advanced-topics/compilation.md#simulation-with-the-virtual-library) to configure the accuracy/speed trade-off for tree-based models and neural networks, using grid-search or using your own heuristics.
{% endhint %}

## List of Examples

### 1. Linear and Logistic Regression

[<img src="../.gitbook/assets/jupyter_logo.png">   Linear Regression example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/LinearRegression.ipynb)
[<img src="../.gitbook/assets/jupyter_logo.png">   Logistic Regression example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/LogisticRegression.ipynb)

These examples show how to use the built-in linear models on synthetic data, which allows easy visualization of the decision boundaries or trend lines. Executing these 1-d and 2-d models in FHE takes around 1 millisecond.

### 2. Generalized Linear Models

[<img src="../.gitbook/assets/jupyter_logo.png">   Poisson Regression example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/PoissonRegression.ipynb)
[<img src="../.gitbook/assets/jupyter_logo.png">   Generalized Linear Models comparison](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/GLMComparison.ipynb)

These two examples show generalized linear models (GLM) on the  real-world [OpenML insurance](https://www.openml.org/d/41214) data-set. As the non-linear inverse-link functions are computed, these models do not use [PBS](../getting-started/concepts.md#cryptography-concepts), and are, thus, very fast (~1ms execution time).

### 3. Decision Tree

[<img src="../.gitbook/assets/jupyter_logo.png">    Decision Tree example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/DecisionTreeClassifier.ipynb)

Using the [OpenML spams](https://www.openml.org/d/44) data-set, this example shows how to train a classifier that detects spam, based on features extracted from e-mail messages. A grid-search is performed over decision-tree hyper-parameters to find the best ones.

### 4. XGBoost and Random Forest Classifier

[<img src="../.gitbook/assets/jupyter_logo.png">   XGBoost/Random Forest example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/XGBClassifier.ipynb)

This example shows how to train tree-ensemble models (either XGBoost or Random Forest), first on a synthetic data-set, and, then, on the [Diabetes](https://www.openml.org/d/37) data-set. Grid search is used to find the best number of trees in the ensemble.

### 5. XGBoost Regression

[<img src="../.gitbook/assets/jupyter_logo.png">   XGBoost Regression example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/XGBRegressor.ipynb)

Privacy-preserving prediction of house prices is shown in this example, using the [House Prices](https://www.openml.org/d/43926) data-set. Using 50 trees in the ensemble, with 5 bits of precision for the input features, the FHE regressor obtains an $$R^2$$ score of 0.90, and an execution time of 7-8 seconds.

### 6. Fully Connected Neural Network

[<img src="../.gitbook/assets/jupyter_logo.png">   NN Iris example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/FullyConnectedNeuralNetwork.ipynb)
[<img src="../.gitbook/assets/jupyter_logo.png">   NN MNIST example](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb)

Two different configurations of the built-in fully-connected neural networks are shown. First, a small bit-width accumulator network is trained on [Iris](https://www.openml.org/d/61) and compared to a Pytorch floating point network. Second, a larger accumulator (>8 bits) is demonstrated on [MNIST](http://yann.lecun.com/exdb/mnist/).

### 7. Comparison of classifiers

[<img src="../.gitbook/assets/jupyter_logo.png">   Classifier comparison](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/ClassifierComparison.ipynb)

Based on three different synthetic data-sets, all the built-in classifiers are demonstrated in this notebook, showing accuracies, inference times, accumulator bit-widths and decision boundaries.
