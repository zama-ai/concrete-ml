# Nearest neighbors

Concrete ML offers nearest neighbors non-parametric classification models with a scikit-learn interface through the `KNeighborsClassifier` class.

|                                              Concrete ML                                              | scikit-learn                                                                                                          |
| :---------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------- |
| [KNeighborsClassifier](../references/api/concrete.ml.sklearn.neighbors.md#class-kneighborsclassifier) | [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |

## Example usage

```python
from concrete.ml.sklearn import KNeighborsClassifier

concrete_classifier = KNeighborsClassifier(n_bits=2, n_neighbors=3)
```

The `KNeighborsClassifier` class quantizes the training data-set that is given to `.fit` with the specified number of bits, `n_bits`. As this value must be kept low to comply with [accumulator size constraints](../getting-started/concepts.md#model-accuracy-considerations-under-fhe-constraints) the accuracy of the model will depend heavily a well-chosen value `n_bits` and the dimensionality of the data.

The `predict` method of the `KNeighborsClassifier` performs the following steps:

- quantization of the test vectors, performed in the clear
- computation of the top-k class indices of the closest training set vector, on encrypted data
- majority vote of the top-k class labels to find the class for each test vector, performed in the clear

## Inference time considerations

The FHE inference latency of this model is heavily influenced by the `n_bits`, the dimensionality of the data. Furthermore, the size of the data-set has a linear impact on the complexity of the data and the number of nearest neighbors, `n_neighbors`, also plays a role.

The KNN computation executes in FHE in $$O(Nlog^2k)$$ steps, where $$N$$ is the training data-set size and $$k$$ is `n_neighbors`. Each step requires several PBS, but the run-time of each of these PBS is influenced by the factors listed above. These factors combine to give the precision required to represent the distances between test vectors and the training data-set vectors. The PBS input precision required by the circuit is related to the one of the distance values.
