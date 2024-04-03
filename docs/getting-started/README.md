# What is Concrete ML?

<figure><img src="../.gitbook/assets/doc_header_CML.png" alt=""><figcaption></figcaption></figure>

Concrete ML is an open source, privacy-preserving, machine learning framework based on Fully Homomorphic Encryption (FHE). It enables data scientists without any prior knowledge of cryptography to:

- automatically turn machine learning models into their FHE equivalent, using familiar APIs from scikit-learn and PyTorch (see how this works for [linear models](../built-in-models/linear.md), [tree-based models](../built-in-models/tree.md), and [neural networks](../built-in-models/neural-networks.md)).
- [train models](../built-in-models/training.md) on encrypted data.
- [pre-process encrypted data](../built-in-models/encrypted_dataframe.md) through a data-frame paradigm

Fully Homomorphic Encryption is an encryption technique that allows computing directly on encrypted data, without needing to decrypt it. With FHE, you can build private-by-design applications without compromising on features. You can learn more about FHE in [this introduction](https://www.zama.ai/post/tfhe-deep-dive-part-1) or by joining the [FHE.org](https://fhe.org) community.

Training on encrypted data provides the highest level of privacy but is slower than training on clear data. Federated learning is an alternative approach, where data privacy can be ensured by using a trusted gradient aggregator, coupled with optional _differential privacy_ instead of encryption. Concrete ML can import linear models, including logistic regression, that are trained using federated learning using the [`from_sklearn` function](../built-in-models/linear.md#pre-trained-models).

## Example usage

Here is a simple example of classification on encrypted data using logistic regression. More examples can be found [here](../tutorials/ml_examples.md).

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression

# Lets create a synthetic data-set
x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)

# Split the data-set into a train and test set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Now we train in the clear and quantize the weights
model = LogisticRegression(n_bits=8)
model.fit(X_train, y_train)

# We can simulate the predictions in the clear
y_pred_clear = model.predict(X_test)

# We then compile on a representative set
model.compile(X_train)

# Finally we run the inference on encrypted inputs
y_pred_fhe = model.predict(X_test, fhe="execute")

print(f"In clear  : {y_pred_clear}")
print(f"In FHE    : {y_pred_fhe}")
print(f"Similarity: {(y_pred_fhe == y_pred_clear).mean():.1%}")

# Output:
    # In clear  : [0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0]
    # In FHE    : [0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0]
    # Similarity: 100.0%
```

It is also possible to call encryption, model prediction, and decryption functions separately as follows. Executing these steps separately is equivalent to calling `predict_proba` on the model instance.

<!--pytest-codeblocks:cont-->

```python
# Predict probability for a single example
y_proba_fhe = model.predict_proba(X_test[[0]], fhe="execute")

# Quantize an original float input
q_input = model.quantize_input(X_test[[0]])

# Encrypt the input
q_input_enc = model.fhe_circuit.encrypt(q_input)

# Execute the linear product in FHE 
q_y_enc = model.fhe_circuit.run(q_input_enc)

# Decrypt the result (integer)
q_y = model.fhe_circuit.decrypt(q_y_enc)

# De-quantize and post-process the result
y0 = model.post_processing(model.dequantize_output(q_y))

print("Probability with `predict_proba`: ", y_proba_fhe)
print("Probability with encrypt/run/decrypt calls: ", y0)
```

This example shows the typical flow of a Concrete ML model:

- The model is trained on unencrypted (plaintext) data using scikit-learn. As FHE operates over integers, Concrete ML quantizes the model to use only integers during inference.
- The quantized model is compiled to an FHE equivalent. Under the hood, the model is first converted to a Concrete Python program, then compiled.
- Inference can then be done on encrypted data. The above example shows encrypted inference in the model-development phase. Alternatively, during [deployment](cloud.md) in a client/server setting, the data is encrypted by the client, processed securely by the server, and then decrypted by the client.

## Current limitations

To make a model work with FHE, the only constraint is to make it run within the supported precision limitations of Concrete ML (currently 16-bit integers). Thus, machine learning models must be quantized, which sometimes leads to a loss of accuracy versus the original model, which operates on plaintext.

Additionally, Concrete ML currently only supports training on encrypted data for some models, while it supports _inference_ for a large variety of models.

Finally, there is currently no support for pre-processing model inputs and post-processing model outputs. These processing stages may involve text-to-numerical feature transformation, dimensionality reduction, KNN or clustering, featurization, normalization, and the mixing of results of ensemble models.

These issues are currently being addressed, and significant improvements are expected to be released in the near future.

## Concrete stack

Concrete ML is built on top of Zama's [Concrete](https://github.com/zama-ai/concrete).

## Online demos and tutorials

Various tutorials are available for [built-in models](../tutorials/ml_examples.md) and [deep learning](../tutorials/dl_examples.md). Several stand-alone demos for use cases can be found in the [Demos and Tutorials](../tutorials/showcase.md) section.

If you have built awesome projects using Concrete ML, feel free to let us know and we'll link to your work!

## Additional resources

- [Dedicated Concrete ML community support](https://community.zama.ai/c/concrete-ml/8)
- [Zama's blog](https://www.zama.ai/blog)
- [FHE.org community](https://fhe.org)

## Support

- Support forum: [https://community.zama.ai](https://community.zama.ai) (we answer in less than 24 hours).
- Live discussion on the FHE.org Discord server: [https://discord.fhe.org](https://discord.fhe.org) (inside the #**concrete** channel).
- Do you have a question about Zama? Write us on [Twitter](https://twitter.com/zama_fhe) or send us an email at: **hello@zama.ai**
