# Built-in Model Examples

The following table summarizes the various examples in this section, along with their accuracies.

| Model                 | Data-set                                                  | Metric                            | Floating Point | Simulation | FHE      |
| --------------------- | --------------------------------------------------------- | --------------------------------- | -------------- | ---------- | -------- |
| Linear Regression     | Synthetic 1D                                              | R2                                | 0.90           | 0.90       | 0.90     |
| Logistic Regression   | Synthetic 2D with 2 classes                               | accuracy                          | 0.92           | 0.92       | 0.92     |
| Poisson Regression    | [OpenML insurance (freq)](https://www.openml.org/d/41214) | mean Poisson deviance             | 1.21           | 1.21       | 1.21     |
| Gamma Regression      | [OpenML insurance (sev)](https://www.openml.org/d/41215)  | mean Gamma deviance               | 0.33           | 0.33       | 0.33     |
| Tweedie Regression    | [OpenML insurance (sev)](https://www.openml.org/d/41215)  | mean Tweedie deviance (power=1.9) | 38.55          | 38.55      | 38.55    |
| Decision Tree         | [OpenML spams](https://www.openml.org/d/44)               | precision score                   | 0.95           | 0.97       | 0.97\*   |
| XGBoost Classifier    | [Diabetes](https://www.openml.org/d/37)                   | MCC                               | 0.48           | 0.52       | 0.52\*   |
| XGBoost Regressor     | [House Prices](https://www.openml.org/d/43926)            | R2                                | 0.92           | 0.90       | 0.90\*   |
| (Built-in) Neural Net | [Iris](https://www.openml.org/d/61)                       | accuracy                          | N/A            | 0.89       | 0.89     |
| (Built-in) Neural Net | [MNIST](http://yann.lecun.com/exdb/mnist/)                | accuracy                          | N/A            | 0.96       | 0.96\*\* |

_A * means that the metric was calculated on a subset of the validation set._

_A \*\* means that the metric was calculated using FHE simulation._

## Concrete-ML built-in models

- [Linear Regression](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/LinearRegression.ipynb)
- [Logistic Regression](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/LogisticRegression.ipynb)
- [Poisson Regression on Risk Features](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/PoissonRegression.ipynb)
- [Decision Tree on Spam Classification ](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/DecisionTreeClassifier.ipynb)
- [XGBoost Classification on Diabetes Detection](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/XGBClassifier.ipynb)
- [XGBoost Regression on House Prices](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/XGBRegressor.ipynb)
- [Generalized Linear Models Comparison on Risk Features](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/GLMComparison.ipynb)
- [Fully-Connected Neural Network on Iris](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/FullyConnectedNeuralNetwork.ipynb)
- [Fully-Connected Neural Network on MNIST](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb)

## Comparison of classifiers

- [Classifier Comparison](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/ClassifierComparison.ipynb)
