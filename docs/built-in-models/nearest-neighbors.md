# Nearest neighbors

This document introduces the nearest neighbors non-parametric classification models that Concrete ML provides with a scikit-learn interface through the `KNeighborsClassifier` class.

|                                              Concrete ML                                              | scikit-learn                                                                                                          |
| :---------------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------- |
| [KNeighborsClassifier](../references/api/concrete.ml.sklearn.neighbors.md#class-kneighborsclassifier) | [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |

## Example

```python
from concrete.ml.sklearn import KNeighborsClassifier

concrete_classifier = KNeighborsClassifier(n_bits=2, n_neighbors=3)
```

## Quantization parameters

The `KNeighborsClassifier` class quantizes the training data-set provided to `.fit` using the specified number of bits (`n_bits`). To comply with [accumulator size constraints](../getting-started/concepts.md#model-accuracy-considerations-under-fhe-constraints), you must keep this value low. The model's accuracy will depend significantly on a well-chosen `n_bits` value and the dimensionality of the data.

The `predict` method of the `KNeighborsClassifier` performs the following steps:

1. Quantize the test vectors on clear data
1. Compute the top-k class indices of the closest training set vector on encrypted data
1. Vote for the top-k class labels to find the class for each test vector, performed on clear data

## Inference time considerations

The FHE inference latency of this model is heavily influenced by the `n_bits` and the dimensionality of the data. Additionally, the data-set size has a linear impact on the data complexity. The number of nearest neighbors (`n_neighbors`) also affects performance.

The KNN computation executes in FHE in $$O(Nlog^2k)$$ steps, where $$N$$ is the training data-set size and $$k$$ is `n_neighbors`. Each step requires several [PBS operations](../getting-started/concepts.md#cryptography-concepts), with their runtime affected by the factors listed above. These factors determine the precision needed to represent the distances between test vectors and training data-set vectors. The PBS input precision required by the circuit is related to the precision of the distance values.
