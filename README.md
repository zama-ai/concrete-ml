<p align="center">
<!-- product name logo -->
  <img width=600 src="https://user-images.githubusercontent.com/5758427/231206363-a6639df8-2253-40d8-836e-e3c22d36d6ad.png">
</p>
<hr>

<p align="center">
  <a href="https://docs.zama.ai/concrete-ml"> 📒 Read documentation</a> | <a href="https://zama.ai/community"> 💛 Community support</a>
</p>
<p align="center">
<!-- Version badge using shields.io -->
  <a href="https://github.com/zama-ai/concrete-ml/releases">
    <img src="https://img.shields.io/github/v/release/zama-ai/concrete-ml?style=flat-square">
  </a>
<!-- Link to tutorials badge using shields.io -->
  <a href="#online-demos-and-tutorials">
    <img src="https://img.shields.io/badge/Learn-%20Tutorials%20and%20demos-orange?style=flat-square">
  </a>
<!-- Zama Bounty Program -->
  <a href="https://github.com/zama-ai/bounty-program">
    <img src="https://img.shields.io/badge/Contribute-Zama%20Bounty%20Program-yellow?style=flat-square">
  </a>
</p>
<hr>

Concrete ML is a Privacy-Preserving Machine Learning (PPML) open-source set of tools built on top of [Concrete](https://github.com/zama-ai/concrete) by [Zama](https://github.com/zama-ai). It aims to simplify the use of fully homomorphic encryption (FHE) for data scientists to help them automatically turn machine learning models into their homomorphic equivalent. Concrete ML was designed with ease-of-use in mind, so that data scientists can use it without knowledge of cryptography. Notably, the Concrete ML model classes are similar to those in scikit-learn and it is also possible to convert PyTorch models to FHE.

## Main features.

Data scientists can use models with APIs which are close to the frameworks they use, with additional options to run inferences in FHE.

Concrete ML features:

- built-in models, which are ready-to-use FHE-friendly models with a user interface that is equivalent to their the scikit-learn and XGBoost counterparts
- support for customs models that can use quantization aware training. These are developed by the user using PyTorch or keras/tensorflow and are imported into Concrete ML through ONNX

## Installation.

Depending on your OS, Concrete ML may be installed with Docker or with pip:

|                 OS / HW                 | Available on Docker | Available on pip |
| :-------------------------------------: | :-----------------: | :--------------: |
|                  Linux                  |         Yes         |       Yes        |
|                 Windows                 |         Yes         |   Coming soon    |
|       Windows Subsystem for Linux       |         Yes         |       Yes        |
|            macOS 11+ (Intel)            |         Yes         |       Yes        |
| macOS 11+ (Apple Silicon: M1, M2, etc.) |         Yes         |       Yes        |

Note: Concrete ML only supports Python `3.8`, `3.9` and `3.10`.

Concrete ML can be installed on Kaggle ([see question on community for more details](https://community.zama.ai/t/how-do-we-use-concrete-ml-on-kaggle/332)) and on Google Colab.

### Docker

To install with Docker, pull the `concrete-ml` image as follows:

`docker pull zamafhe/concrete-ml:latest`

### Pip

To install Concrete ML from PyPi, run the following:

```
pip install -U pip wheel setuptools
pip install concrete-ml
```

You can find more detailed installation instructions in [this part of the documentation](docs/getting-started/pip_installing.md)

## A simple Concrete ML example with scikit-learn.

A simple example which is very close to scikit-learn is as follows, for a logistic regression :

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
y_pred_fhe = model.predict(X_test, fhe="execute")

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print(f"Similarity: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")

# Output:
    # In clear  : [0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0]
    # In FHE    : [0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0]
    # Similarity: 100%
```

It is also possible to call encryption, model prediction, and decryption functions separately as follows.
Executing these steps separately is equivalent to calling `predict_proba` on the model instance.

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

This example is explained in more detail in the [linear model documentation](docs/built-in-models/linear.md). Concrete ML built-in models
have APIs that are almost identical to their scikit-learn counterparts. It is also possible to convert PyTorch networks to FHE with the Concrete ML conversion APIs. Please refer to the [linear models](docs/built-in-models/linear.md), [tree-based models](docs/built-in-models/tree.md) and [neural networks](docs/built-in-models/neural-networks.md) documentation for more examples, showing the scikit-learn-like API of the built-in
models.

## Documentation.

Full, comprehensive documentation is available here: [https://docs.zama.ai/concrete-ml](https://docs.zama.ai/concrete-ml).

## Online demos and tutorials.

Various tutorials are given for [built-in models](docs/built-in-models/ml_examples.md) and for [deep learning](docs/deep-learning/examples.md) In addition, several complete use-cases are explored:

- [Encrypted Large Language Model](use_case_examples/llm/): convert a user-defined part of a Large Language Model for encrypted text generation. Shows the trade-off between quantization and accuracy for text generation and shows how to run the model in FHE.

- [Credit Scoring](use_case_examples/credit_scoring/): predict the chance of a given loan applicant defaulting on loan repayment while keeping the user's data private. Shows how Concrete ML models easily replace their scikit-learn equivalents

- [Health diagnosis](use_case_examples/disease_prediction/): based on a patient's symptoms, history and other health factors, give
  a diagnosis using FHE to preserve the privacy of the patient.

- [Titanic](use_case_examples/titanic/KaggleTitanic.ipynb): solve the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/). Implemented with XGBoost from Concrete ML, this example comes as a companion of the [Kaggle notebook](https://www.kaggle.com/code/concretemlteam/titanic-with-privacy-preserving-machine-learning), and was the subject of a blogpost in [KDnuggets](https://www.kdnuggets.com/2022/08/machine-learning-encrypted-data.html).

- [Sentiment analysis with transformers](use_case_examples/sentiment_analysis_with_transformer): predict if an encrypted tweet / short message is positive, negative or neutral, using FHE. The [live interactive](https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis) demo is available on Hugging Face. This [blog post](https://huggingface.co/blog/sentiment-analysis-fhe) explains how this demo works!

- [CIFAR10 FHE-friendly model with Brevitas](use_case_examples/cifar/cifar_brevitas_training): train a VGG9 FHE-compatible neural network using Brevitas, and a script to run the neural network in FHE. Execution in FHE takes ~4 minutes per image and shows an accuracy of 88.7%.

- [CIFAR10 / CIFAR100 FHE-friendly models with Transfer Learning approach](use_case_examples/cifar/cifar_brevitas_finetuning): series of three notebooks, that convert a pre-trained FP32 VGG11 neural network into a quantized model using Brevitas. The model is fine-tuned on the CIFAR data-sets, converted for FHE execution with Concrete ML and evaluated using FHE simulation. For CIFAR10 and CIFAR100, respectively, our simulations show an accuracy of 90.2% and 68.2%.

- [FHE neural network splitting for client/server deployment](use_case_examples/cifar/cifar_brevitas_with_model_splitting): explains how to split a computationally-intensive neural network model in two parts. First, we execute the first part on the client side in the clear, and the output of this step is encrypted. Next, to complete the computation, the second part of the model is evaluated with FHE. This tutorial also shows the impact of FHE speed/accuracy trade-off on CIFAR10, limiting PBS to 8-bit, and thus achieving 62% accuracy.

- [Encrypted image filtering](use_case_examples/image_filtering): filter encrypted images by applying filters such as black-and-white, ridge detection, or your own filter.

If you have built awesome projects using Concrete ML, feel free to let us know and we'll link to them!

## Contributing 

To contribute to Concrete ML, please refer to this [section of the documentation](docs/developer-guide/contributing.md).

## Citing Concrete ML

To cite Concrete ML, notably in academic papers, please use the following entry, which list authors by order of first commit:

```text
@Misc{ConcreteML,
  title={Concrete {ML}: a Privacy-Preserving Machine Learning Library using Fully Homomorphic Encryption for Data Scientists},
  author={Zama},
  year={2022},
  note={\url{https://github.com/zama-ai/concrete-ml}},
}
```

## Need support?

<a target="_blank" href="https://community.zama.ai">
  <img src="https://user-images.githubusercontent.com/5758427/231115030-21195b55-2629-4c01-9809-be5059243999.png">
</a>

## License.

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
