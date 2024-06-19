# What is Concrete ML?

<figure><img src="../.gitbook/assets/doc_header_CML.png" alt=""><figcaption></figcaption></figure>

Concrete ML is an open source, privacy-preserving, machine learning framework based on Fully Homomorphic Encryption (FHE). It enables data scientists without any prior knowledge of cryptography to perform:

- **Automatic model conversion**: Use familiar APIs from scikit-learn and PyTorch to convert machine learning models to their FHE equivalent. This is applicable for [linear models](../built-in-models/linear.md), [tree-based models](../built-in-models/tree.md), and [neural networks](../built-in-models/neural-networks.md)).
- **Encrypted data training**: [Train models](../built-in-models/training.md) directly on encrypted data to maintain privacy.
- **Encrypted data pre-processing**: [Pre-process encrypted data](../built-in-models/encrypted_dataframe.md) using a DataFrame paradigm.

## Key features

- **Training on encrypted data**: FHE is an encryption technique that allows computing directly on encrypted data, without needing to decrypt it. With FHE, you can build private-by-design applications without compromising on features. Learn more about FHE in [this introduction](https://www.zama.ai/post/tfhe-deep-dive-part-1) or join the [FHE.org](https://fhe.org) community.

- **Federated learning**: Training on encrypted data provides the highest level of privacy but is slower than training on clear data. Federated learning is an alternative approach, where data privacy can be ensured by using a trusted gradient aggregator, coupled with optional _differential privacy_ instead of encryption. Concrete ML can import all types of models: linear, tree-based and neural networks, that are trained using federated learning using the [`from_sklearn_model` function](../built-in-models/linear.md#pre-trained-models) and the [`compile_torch_model`](../deep-learning/torch_support.md) function.

## Example usage

Here is a simple example of classification on encrypted data using logistic regression. You can find more examples [here](../tutorials/ml_examples.md).

This example shows the typical flow of a Concrete ML model:

1. **Training the model**: Train the model on unencrypted (plaintext) data using scikit-learn. Since Fully Homomorphic Encryption (FHE) operates over integers, Concrete ML quantizes the model to use only integers during inference.
1. **Compiling the model**: Compile the quantized model to an FHE equivalent. Under the hood, the model is first converted to a Concrete Python program and then compiled.
1. **Performing inference**: Perform inference on encrypted data. The example above shows encrypted inference in the model-development phase. Alternatively, during [deployment](cloud.md) in a client/server setting, the client encrypts the data, the server processes it securely, and then the client decrypts the results.

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

## Current limitations

- **Precision and accuracy**: In order to run models in FHE, Concrete ML requires models to be within the precision limit, currently 16-bit integers. Thus, machine learning models must be quantized and it sometimes leads to a loss of accuracy versus the original model that operates on plaintext.

- **Models availability**: Concrete ML currently only supports _training_ on encrypted data for some models, while it supports _inference_ for a large variety of models.

- **Processing**: Concrete currently doesn't support pre-processing model inputs and post-processing model outputs. These processing stages may involve:

  - Text-to-numerical feature transformation
  - Dimensionality reduction
  - KNN or clustering
  - Featurization
  - Normalization
  - The mixing of ensemble models' results.

These issues are currently being addressed, and significant improvements are expected to be released in the near future.

## Concrete stack

Concrete ML is built on top of Zama's [Concrete](https://github.com/zama-ai/concrete).

## Online demos and tutorials

Various tutorials are available for [built-in models](../tutorials/ml_examples.md) and [deep learning](../tutorials/dl_examples.md). Several stand-alone demos for use cases can be found in the [Demos and Tutorials](../tutorials/showcase.md) section.

If you have built awesome projects using Concrete ML, feel free to let us know and we'll link to your work!

## Additional resources

- [Zama's blog](https://www.zama.ai/blog)

## Support

- [Community channels](https://zama.ai/community-channels) (we answer in less than 24 hours).
