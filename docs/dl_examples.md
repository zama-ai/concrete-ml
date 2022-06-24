# Deep Learning Examples

Here is a summary of our results. Remark that from one seed to the other, results of the different notebooks may vary. Please look in the different notebooks for details.

| Model                  | Dataset                                                                                       | Metric   | Clear | Quantized | FHE   |
| ---------------------- | --------------------------------------------------------------------------------------------- | -------- | ----- | --------- | ----- |
| Fully Connected NN     | [Iris](https://www.openml.org/d/61)                                                           | accuracy | 0.947 | 0.895     | 0.895 |
| QAT Fully Connected NN | Synthetic (Checkerboard)                                                                      | accuracy | 0.94  | 0.94      | 0.94  |
| Convolutional NN       | [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | accuracy | 0.90  | \*\*      | \*\*  |

In this table, \*\* means that the accuracy is actually random-like, because the quantization we need to set to fullfill bitsize constraints is too strong.

## Custom models

- [FullyConnectedNeuralNetwork.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/FullyConnectedNeuralNetwork.ipynb)
- [QuantizationAwareTraining.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/QuantizationAwareTraining.ipynb)
- [ConvolutionalNeuralNetwork.ipynb](https://github.com/zama-ai/concrete-ml-internal/tree/main/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb)
