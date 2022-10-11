# What is Concrete ML?

[⭐️ Star the repo on Github](https://github.com/zama-ai/concrete-ml) | [🗣 Community support forum](https://community.zama.ai/c/concrete-ml/8) | [📁 Contribute to the project](https://github.com/zama-ai/concrete-ml/blob/main/docs/dev/howto/contributing.md)

![](.gitbook/assets/3.png)

Concrete-ML is an open-source privacy-preserving machine learning inference framework based on fully homomorphic encryption (FHE). It enables data scientists without any prior knowledge of cryptography to automatically turn machine learning models into their FHE equivalent, using familiar APIs from Scikit-learn and PyTorch (see how it looks for [linear models](built-in-models/linear.md), [tree-based models](built-in-models/tree.md) and [neural networks](built-in-models/neural-networks.md)).

Fully Homomorphic Encryption (FHE) is an encryption technique that allows computing directly on encrypted data, without needing to decrypt it. With FHE, you can build private-by-design applications without compromising on features. You can learn more about FHE in [this introduction](https://www.zama.ai/post/tfhe-deep-dive-part-1), or by joining the [FHE.org](https://fhe.org) community.

## Example usage

Here is a simple example of classification on encrypted data using logistic regression. More examples can be found [here](built-in-models/ml_examples.md).

```python
import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression

# Lets create a synthetic data-set
x, y = make_classification(n_samples=100,
    class_sep=2, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
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

- The model is trained on unencrypted (plaintext) data using scikit-learn. As FHE operates over integers, Concrete-ML quantizes the model to use only integers during inference.
- The quantized model is compiled to a FHE equivalent. Under the hood, the model is first converted to a Concrete-Numpy program, then compiled.
- Inference can then be done on encrypted data. The above example shows encrypted inference in the model development phase. Alternatively, in [deployment](getting-started/cloud.md) in a client/server setting, the data is encrypted by the client, processed securely by the server and then decrypted by the client.

## Current limitations

To make a model work with FHE, the only constraint is to make it run within the supported precision limitations of Concrete-ML (currently 8-bit integers). Thus, machine learning models are required to be quantized, which sometimes leads to a loss of accuracy versus the original model operating on plaintext.

Additionally, Concrete-ML currently only supports FHE _inference_. On the other hand, training has to be done on unencrypted data, producing a model which is then converted to a FHE equivalent that can perform encrypted inference, i.e. prediction over encrypted data.

Finally, in Concrete-ML there is currently no support for pre-processing model inputs and for post-processing model outputs. These processing stages may involve text to numerical feature transformation, dimensionality reduction, KNN or clustering, featurization, normalization, and the mixing of results of ensemble models.

All of these issues are currently being addressed and significant improvements are expected to be released in the coming months.

## Concrete Stack

Concrete-ML is built on top of Zama's Concrete framework. It uses [Concrete-Numpy](https://github.com/zama-ai/concrete-numpy), which itself uses the [Concrete-Compiler](https://pypi.org/project/concrete-compiler) and the [Concrete-Library](https://docs.zama.ai/concrete). To use these libraries directly, refer to the [Concrete-Numpy](https://docs.zama.ai/concrete-numpy/) and [Concrete-Framework](https://docs.zama.ai/concrete) documentations.

## Online demos and tutorials.

Various tutorials are proposed for the [built-in models](built-in-models/ml_examples.md) and for [deep learning](deep-learning/examples.md). In addition, we also list standalone use-cases:

- [MNIST](https://github.com/zama-ai/concrete-ml-internal/blob/main/use%5C_case%5C_examples/mnist/README.md): a Python and notebook showing a Quantization Aware Training (done with [Brevitas](https://github.com/Xilinx/brevitas) and following constraints of the package) and its corresponding use in Concrete-ML.
- [Encrypted sentiment analysis](https://github.com/zama-ai/concrete-ml-internal/blob/main/use%5C_case%5C_examples/encrypted%5C_sentiment%5C_analysis/README.md): a gradio demo, which predicts if a tweet / short message is positive, negative or neutral. Of course, this is completed using FHE. The corresponding application is directly available \[FIXME\] https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis.

More generally, if you have built awesome projects using Concrete-ML, feel free to let us know and we'll link to it!

## Additional resources

- [Dedicated Concrete-ML community support](https://community.zama.ai/c/concrete-ml/8)
- [Zama's blog](https://www.zama.ai/blog)
- [FHE.org community](https://fhe.org)

## Looking for support? Ask our team!

- Support forum: [https://community.zama.ai](https://community.zama.ai) (we answer in less than 24 hours).
- Live discussion on the FHE.org discord server: [https://discord.fhe.org](https://discord.fhe.org) (inside the #**concrete** channel).
- Do you have a question about Zama? You can write us on [Twitter](https://twitter.com/zama%5C_fhe) or send us an email at: **hello@zama.ai**
