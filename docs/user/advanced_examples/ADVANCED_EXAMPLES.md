# Additional Examples

## **Concrete-ML** models

The following table summarizes the various examples in this section, along with their accuracies.

| Model               | Dataset                                                   | Metric                | Clear | Quantized | FHE    |
| ------------------- | --------------------------------------------------------- | --------------------- | ----- | --------- | ------ |
| Linear Regression   | Synthetic 1D                                              | R2                    | 0.876 | 0.863     | 0.863  |
| Logistic Regression | Synthetic 2D with 2 classes                               | accuracy              | 0.90  | 0.875     | 0.875  |
| Poisson Regression  | [OpenML insurance (freq)](https://www.openml.org/d/41214) | mean Poisson deviance | 1.38  | 1.68      | 1.68   |
| Decision Tree       | [OpenML spams](https://www.openml.org/d/44)               | precision score       | 0.95  | 0.97      | 0.97\* |
| XGBoost             | [Diabetes](https://www.openml.org/d/37)                   | MCC                   | 0.48  | 0.52      | 0.52\* |

_A * means that FHE accuracy was calculated on a subset of the validation set._

- [LinearRegression.ipynb](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/LinearRegression.ipynb)
- [LogisticRegression.ipynb](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/LogisticRegression.ipynb)
- [PoissonRegression.ipynb](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/PoissonRegression.ipynb)
- [DecisionTreeClassifier.ipynb](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/DecisionTreeClassifier.ipynb)
- [XGBClassifier.ipynb](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/XGBClassifier.ipynb)

## Comparison of classifiers

- [ClassifierComparison.ipynb](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/ClassifierComparison.ipynb)

## Deep learning

| Model              | Dataset                                                                                          | Metric   | Clear | Quantized | FHE   |
| ------------------ | ------------------------------------------------------------------------------------------------ | -------- | ----- | --------- | ----- |
| Fully Connected NN | [Iris](https://www.openml.org/d/61)                                                              | accuracy | 0.947 | 0.895     | 0.895 |
| Convolutional NN   | [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load%5C_digits.html) | accuracy | 0.90  | \*\*      | \*\*  |

In this table, \*\* means that the accuracy is actually random-like, because the quantization we need to set to fullfill bitsize constraints is too strong.

- [Fully Connected NN](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/FullyConnectedNeuralNetwork.ipynb)
- [Convolutional NN](https://github.com/zama-ai/concrete-ml/tree/main/docs/user/advanced%5C_examples/ConvolutionalNeuralNetwork.ipynb)
