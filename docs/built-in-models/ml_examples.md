# Built-in Model Examples

The following table summarizes the various examples in this section, along with their accuracies.

| Model                 | Data-set                                                  | Metric                            | Floating Point | Simulation | FHE    |
| --------------------- | --------------------------------------------------------- | --------------------------------- | -------------- | ---------- | ------ |
| Linear Regression     | Synthetic 1D                                              | R2                                | 0.90           | 0.90       | 0.90   |
| Logistic Regression   | Synthetic 2D with 2 classes                               | accuracy                          | 0.92           | 0.92       | 0.92   |
| Poisson Regression    | [OpenML insurance (freq)](https://www.openml.org/d/41214) | mean Poisson deviance             | 1.21           | 1.21       | 1.21   |
| Gamma Regression      | [OpenML insurance (sev)](https://www.openml.org/d/41215)  | mean Gamma deviance               | 0.33           | 0.33       | 0.33   |
| Tweedie Regression    | [OpenML insurance (sev)](https://www.openml.org/d/41215)  | mean Tweedie deviance (power=1.9) | 38.55          | 38.55      | 38.55  |
| Decision Tree         | [OpenML spams](https://www.openml.org/d/44)               | precision score                   | 0.95           | 0.97       | 0.97\* |
| XGBoost Classifier    | [Diabetes](https://www.openml.org/d/37)                   | MCC                               | 0.48           | 0.52       | 0.52\* |
| XGBoost Regressor     | [House Prices](https://www.openml.org/d/43926)            | R2                                | 0.92           | 0.90       | 0.90\* |
| (Built-in) Neural Net | [MNIST](http://yann.lecun.com/exdb/mnist/)                | accuracy                          | N/A            | 0.965      | 0.96\* |

_A * means that FHE accuracy was calculated on a subset of the validation set._

## Concrete-ML models

- [LinearRegression.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/LinearRegression.ipynb)
- [LogisticRegression.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/LogisticRegression.ipynb)
- [PoissonRegression.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/PoissonRegression.ipynb)
- [DecisionTreeClassifier.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/DecisionTreeClassifier.ipynb)
- [XGBClassifier.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/XGBClassifier.ipynb)
- [XGBRegressor.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/XGBRegressor.ipynb)
- [GLMComparison.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/GLMComparison.ipynb)
- [FullyConnectedNeuralNetworkOnMNIST.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb)

## Comparison of classifiers

- [ClassifierComparison.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/ClassifierComparison.ipynb)
