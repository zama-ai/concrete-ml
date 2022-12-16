# What is Concrete ML?

[⭐️ Star the repo on Github](https://github.com/zama-ai/concrete-ml) | [🗣 Community support forum](https://community.zama.ai/c/concrete-ml/8) | [📁 Contribute to the project](developer-guide/contributing.md)

![](.gitbook/assets/3.png)

Concrete-ML is an open-source privacy-preserving machine learning inference framework based on fully homomorphic encryption (FHE). It enables data scientists without any prior knowledge of cryptography to automatically turn machine learning models into their FHE equivalent, using familiar APIs from Scikit-learn and PyTorch (see how it looks for [linear models](built-in-models/linear.md), [tree-based models](built-in-models/tree.md) and [neural networks](built-in-models/neural-networks.md)).

Fully Homomorphic Encryption (FHE) is an encryption technique that allows computing directly on encrypted data, without needing to decrypt it. With FHE, you can build private-by-design applications without compromising on features. You can learn more about FHE in [this introduction](https://www.zama.ai/post/tfhe-deep-dive-part-1), or by joining the [FHE.org](https://fhe.org) community.

## Example usage

Here is a simple example of classification on encrypted data using logistic regression. More examples can be found [here](built-in-models/ml_examples.md).

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

# Finally we run the inference on encrypted inputs !
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print(f"Similarity: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")

# Output:
    # In clear  : [0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0]
    # In FHE    : [0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0]
    # Similarity: 100%
```

This example shows the typical flow of a Concrete-ML model:

- The model is trained on unencrypted (plaintext) data using scikit-learn. As FHE operates over integers, Concrete-ML quantizes the model to use only integers during inference.
- The quantized model is compiled to a FHE equivalent. Under the hood, the model is first converted to a Concrete-Numpy program, then compiled.
- Inference can then be done on encrypted data. The above example shows encrypted inference in the model development phase. Alternatively, in [deployment](getting-started/cloud.md) in a client/server setting, the data is encrypted by the client, processed securely by the server and then decrypted by the client.

## Current limitations

To make a model work with FHE, the only constraint is to make it run within the supported precision limitations of Concrete-ML (currently 16-bit integers). Thus, machine learning models are required to be quantized, which sometimes leads to a loss of accuracy versus the original model operating on plaintext.

Additionally, Concrete-ML currently only supports FHE _inference_. On the other hand, training has to be done on unencrypted data, producing a model which is then converted to a FHE equivalent that can perform encrypted inference, i.e. prediction over encrypted data.

Finally, in Concrete-ML there is currently no support for pre-processing model inputs and post-processing model outputs. These processing stages may involve text-to-numerical feature transformation, dimensionality reduction, KNN or clustering, featurization, normalization, and the mixing of results of ensemble models.

All of these issues are currently being addressed and significant improvements are expected to be released in the coming months.

## Concrete Stack

Concrete-ML is built on top of Zama's Concrete framework. It uses [Concrete-Numpy](https://github.com/zama-ai/concrete-numpy), which itself uses the [Concrete-Compiler](https://pypi.org/project/concrete-compiler) and the [Concrete-Library](https://docs.zama.ai/concrete). To use these libraries directly, refer to the [Concrete-Numpy](https://docs.zama.ai/concrete-numpy/) and [Concrete-Framework](https://docs.zama.ai/concrete) documentations.

## Online demos and tutorials.

Various tutorials are proposed for the [built-in models](built-in-models/ml_examples.md) and for [deep learning](deep-learning/examples.md). In addition, we also list standalone use-cases:

- [MNIST](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/mnist): a Python script and notebook showing quantization-aware training following FHE constraints. A fully-connected neural network is implemented with [Brevitas](https://github.com/Xilinx/brevitas) and is converted to FHE with Concrete-ML.

- [Titanic](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/titanic/KaggleTitanic.ipynb): a notebook, which gives a solution to the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/). Implemented with XGBoost from Concrete-ML, this example comes as a companion of the [Kaggle notebook](https://www.kaggle.com/code/concretemlteam/titanic-with-privacy-preserving-machine-learning), and was the subject of a blogpost in [KDnuggets](https://www.kdnuggets.com/2022/08/machine-learning-encrypted-data.html).

- [Sentiment analysis with transformers](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/sentiment-analysis-with-transformer): a gradio demo which predicts if a tweet / short message is positive, negative or neutral, with FHE of course! The [live interactive](https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis) demo is available on Hugging Face. This [blog post](https://huggingface.co/blog/sentiment-analysis-fhe) explains how this demo works!

- [CIFAR10 FHE-friendly model with Brevitas](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/cifar_brevitas_training): code for training from scratch a VGG-like FHE-compatible neural network using Brevitas, and a script to run the neural network in FHE. FHE simulation shows an accuracy of 88.7%, but running inference with FHE is still a work-in-progress.

- [CIFAR10 / CIFAR100 FHE-friendly models with Transfer Learning approach](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/cifar_brevitas_finetuning): series of three notebooks, that show how to convert a pre-trained FP32 VGG11 neural network into a quantized model using Brevitas. The model is fine-tuned on the CIFAR datasets, converted for FHE execution with Concrete-ML and evaluated using FHE simulation. For CIFAR10 and CIFAR100, respectively, our simulations show an accuracy of 90.2% and 68.2%. True FHE inference is a work-in-progress.

- [FHE neural network splitting for client/server deployment](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/cifar_10_with_model_splitting): we explain how to split a computationally-intensive neural network model in two parts. First, we execute the first part on the client side in the clear, and the output of this step is encrypted. Next, to complete the computation, the second part of the model is evaluated with FHE. This tutorial also shows the impact of FHE speed/accuracy tradeoff on CIFAR10, limiting PBS to 8-bit, and thus achieving 62% accuracy.

- [Encrypted image filtering](https://github.com/zama-ai/concrete-ml-internal/blob/main/use_case_examples/image_filtering): finally, the live demo for our [6-min](https://6min.zama.ai) is available, in the form of a gradio application. We take encrypted images, and apply some filters (for example black-and-white, ridge detection, or your own filter).
  More generally, if you have built awesome projects using Concrete-ML, feel free to let us know and we'll link to it!

More generally, if you have built awesome projects using Concrete-ML, feel free to let us know and we'll link to it!

## Additional resources

- [Dedicated Concrete-ML community support](https://community.zama.ai/c/concrete-ml/8)
- [Zama's blog](https://www.zama.ai/blog)
- [FHE.org community](https://fhe.org)

## Looking for support? Ask our team!

- Support forum: [https://community.zama.ai](https://community.zama.ai) (we answer in less than 24 hours).
- Live discussion on the FHE.org discord server: [https://discord.fhe.org](https://discord.fhe.org) (inside the #**concrete** channel).
- Do you have a question about Zama? You can write us on [Twitter](https://twitter.com/zama_fhe) or send us an email at: **hello@zama.ai**
