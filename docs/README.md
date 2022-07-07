# What is Concrete ML?

<mark style="background-color:yellow;"></mark>[<mark style="background-color:yellow;">⭐️ Star the repo on Github</mark>](https://github.com/zama-ai/concrete-ml) <mark style="background-color:yellow;">|</mark> [<mark style="background-color:yellow;">🗣</mark> <mark style="background-color:yellow;">Community support forum</mark>](https://community.zama.ai/c/concrete-ml/8) <mark style="background-color:yellow;">|</mark> [<mark style="background-color:yellow;">📁</mark> <mark style="background-color:yellow;">Contribute to the project</mark>](https://github.com/zama-ai/concrete-ml/blob/main/docs/dev/howto/contributing.md)<mark style="background-color:yellow;"></mark>

![](.gitbook/assets/zama_docs_intro.jpg)

Concrete-ML is an open-source private machine learning inference framework based on fully homomorphic encryption (FHE). It enables data scientists without any prior knowledge of cryptography to automatically turn machine learning models into their FHE equivalent, using familiar APIs from Scikit-learn and PyTorch (see how it looks for [linear models](built-in-models/linear.md), [tree-based models](built-in-models/tree.md) and [neural networks](built-in-models/neural-networks.md)).

Fully Homomorphic Encryption (FHE) is an encryption technique that allows computating directly on encrypted data, without needing to decrypt it. With FHE, you can build private-by-design applications without compromising on features. You can learn more about FHE in [this introduction](https://www.zama.ai/post/tfhe-deep-dive-part-1), or by joining the [FHE.org](https://fhe.org) community.

## Example usage

Here is a simple example of encrypted inference using logistic regression. More examples can be found [here](built-in-models/ml_examples.md).

<!--
```python
N_TEST = 1
```
-->

<!--pytest-codeblocks:cont-->

```python
import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression

# Lets create a synthetic dataset
N_EXAMPLE_TOTAL = 100
N_TEST = 20 if not 'N_TEST' in locals() else N_TEST
x, y = make_classification(n_samples=N_EXAMPLE_TOTAL,
    class_sep=2, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=N_TEST / N_EXAMPLE_TOTAL, random_state=42
)

# Now we train in plaintext using quantization
model = LogisticRegression(n_bits=2)
model.fit(X_train, y_train)

y_pred_clear = model.predict(X_test)

# Finally we compile and run inference on encrypted inputs!
model.compile(x)
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print("Comparison:", (y_pred_fhe == y_pred_clear))
# Output:
#   In clear  : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   In FHE    : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   Comparison: [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True]
```

This example shows the typical flow of a Concrete-ML model:

- The model is trained on unencrypted (plaintext) data
- The resulting model is quantized to small integers using either post-training quantization or quantization-aware training
- The quantized model is compiled to a FHE equivalent (under the hood, the model is first converted to a Concrete-Numpy program, then compiled)
- Inference can then be done on encrypted data

To make a model work with FHE, the only constrain is to make it run within the supported precision limitations of Concrete-ML (currently 8-bit integers).

## Current Limitations

Concrete-ML is built on top of Zama’s Concrete framework. It uses [Concrete-Numpy](https://github.com/zama-ai/concrete-numpy), which itself uses the [Concrete-Compiler](https://pypi.org/project/concrete-compiler) and the [Concrete-Library](https://docs.zama.ai/concrete-core). To use these libraries directly, refer to the [Concrete-Numpy](https://docs.zama.ai/concrete-numpy/) and [Concrete-Framework](https://docs.zama.ai/concrete) documentations.

Currently, Concrete only supports 8-bit encrypted integer arithmetics. This requires models to be quantized heavily, which sometimes leads to loss of accuracy vs the plaintext model. Furthermore, the Concrete-Compiler is still a work in progress, meaning it won't always find optimal performance parameters, leading to slower than expected execution times.

Additionally, Concrete-ML currently only supports FHE inference. Training on the other hand has to be done on unencrypted data, producing a model which is then converted to an FHE equivalent that can do encrypted inference.

Finally, there is currently no support for pre and post processing in FHE. Data must arrive to the FHE model already pre-processed and post-processing (if there is any) has to be done client-side.

All of these issues are currently being addressed and significant improvements are expected to be released in the coming months.

## Additional resources

- [Concrete-ML community support](https://community.zama.ai/c/concrete-ml/8)
- [Zama's blog](https://www.zama.ai/blog)
- [FHE.org community](https://fhe.org)

## Looking for support? Ask our team!

![](figures/support_zama.jpg)






