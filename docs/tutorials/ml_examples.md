# Built-in model examples

These examples illustrate the basic usage of built-in Concrete ML models. For more examples showing how to train high-accuracy models on more complex data-sets, see the [Demos and Tutorials](showcase.md) section.

## FHE constraints

In Concrete ML, built-in linear models are exact equivalents to their scikit-learn counterparts. As they do not apply any non-linearity during inference, these models are very fast (~1ms FHE inference time) and can use high-precision integers (between 20-25 bits).

Tree-based models apply non-linear functions that enable comparisons of inputs and trained thresholds. Thus, they are limited with respect to the number of bits used to represent the inputs. But as these examples show, in practice 5-6 bits are sufficient to exactly reproduce the behavior of their scikit-learn counterpart models.

In the examples below, built-in neural networks can be configured to work with user-specified accumulator sizes, which allow the user to adjust the speed/accuracy trade-off.

{% hint style="info" %}
It is recommended to use [simulation](../explanations/compilation.md#fhe-simulation) to configure the speed/accuracy trade-off for tree-based models and neural networks, using grid-search or your own heuristics.
{% endhint %}

## List of examples

### 1. Linear models

- [Linear Regression example](../advanced_examples/LinearRegression.ipynb)
- [Logistic Regression example](../advanced_examples/LogisticRegression.ipynb)
- [Linear Support Vector Regression example](../advanced_examples/LinearSVR.ipynb)
- [Linear SVM classification](../advanced_examples/SVMClassifier.ipynb)

These examples show how to use the built-in linear models on synthetic data, which allows for easy visualization of the decision boundaries or trend lines. Executing these 1D and 2D models in FHE takes around 1 millisecond.

### 2. Generalized linear models

- [Poisson Regression example](../advanced_examples/PoissonRegression.ipynb)
- [Generalized Linear Models comparison](../advanced_examples/GLMComparison.ipynb)

These two examples show generalized linear models (GLM) on the real-world [OpenML insurance](https://www.openml.org/d/41214) data-set. As the non-linear, inverse-link functions are computed, these models do not use [PBS](../getting-started/concepts.md#cryptography-concepts), and are, thus, very fast (~1ms execution time).

### 3. Decision tree

- [Decision Tree Classifier](../advanced_examples/DecisionTreeClassifier.ipynb)

Using the [OpenML spams](https://www.openml.org/d/44) data-set, this example shows how to train a classifier that detects spam, based on features extracted from email messages. A grid-search is performed over decision-tree hyper-parameters to find the best ones.

- [Decision Tree Regressor](../advanced_examples/DecisionTreeRegressor.ipynb)

Using the [House Price prediction](https://www.openml.org/search?type=data&sort=runs&id=537) data-set, this example shows how to train regressor that predicts house prices.

### 4. XGBoost and Random Forest classifier

- [XGBoost/Random Forest example](../advanced_examples/XGBClassifier.ipynb)

This example shows how to train tree-ensemble models (either XGBoost or Random Forest), first on a synthetic data-set, and then on the [Diabetes](https://www.openml.org/d/37) data-set. Grid-search is used to find the best number of trees in the ensemble.

### 5. XGBoost regression

- [XGBoost Regression example](../advanced_examples/XGBRegressor.ipynb)

Privacy-preserving prediction of house prices is shown in this example, using the [House Prices](https://www.openml.org/d/43926) data-set. Using 50 trees in the ensemble, with 5 bits of precision for the input features, the FHE regressor obtains an $$R^2$$ score of 0.90 and an execution time of 7-8 seconds.

### 6. Fully connected neural network

- [NN Iris example](../advanced_examples/FullyConnectedNeuralNetwork.ipynb)
- [NN MNIST example](../advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb)

Two different configurations of the built-in, fully-connected neural networks are shown. First, a small bit-width accumulator network is trained on [Iris](https://www.openml.org/d/61) and compared to a PyTorch floating point network. Second, a larger accumulator (>8 bits) is demonstrated on [MNIST](http://yann.lecun.com/exdb/mnist/).

### 7. Comparison of models

- [Classifier comparison](../advanced_examples/ClassifierComparison.ipynb)
- [Regressor comparison](../advanced_examples/RegressorComparison.ipynb)

Based on three different synthetic data-sets, all the built-in classifiers are demonstrated in this notebook, showing accuracies, inference times, accumulator bit-widths, and decision boundaries.
