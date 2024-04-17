<p align="center">
<!-- product name logo -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/zama-ai/concrete-ml/assets/157474013/5ed658d7-0abd-4444-9063-99d8b76c2602">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/zama-ai/concrete-ml/assets/157474013/7c67e594-5e2c-483e-858f-ce473a36e37f">
  <img width=600 alt="Zama Concrete ML">
</picture>
</p>

<hr>

<p align="center">
  <a href="https://docs.zama.ai/concrete-ml"> 📒 Documentation</a> | <a href="https://zama.ai/community"> 💛 Community support</a> | <a href="https://github.com/zama-ai/awesome-zama"> 📚 FHE resources by Zama</a>
</p>

<p align="center">
  <a href="https://github.com/zama-ai/concrete-ml/releases"><img src="https://img.shields.io/github/v/release/zama-ai/concrete-ml?style=flat-square"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause--Clear-%23ffb243?style=flat-square"></a>
  <a href="https://github.com/zama-ai/bounty-program"><img src="https://img.shields.io/badge/Contribute-Zama%20Bounty%20Program-%23ffd208?style=flat-square"></a>
</p>

## About

### What is Concrete ML

**Concrete ML** is a Privacy-Preserving Machine Learning (PPML) open-source set of tools built on top of [Concrete](https://github.com/zama-ai/concrete) by [Zama](https://github.com/zama-ai).

It simplifies the use of fully homomorphic encryption (FHE) for data scientists so that they can automatically turn machine learning models into their homomorphic equivalents, and use them without knowledge of cryptography.

Concrete ML is designed with ease of use in mind. Data scientists can use models with APIs that are close to the frameworks they already know well, while additional options to those models allow them to run inference or training on encrypted data with FHE. The Concrete ML model classes are similar to those in scikit-learn and it is also possible to convert PyTorch models to FHE.
<br></br>

### Main features

- **Built-in models**: Ready-to-use FHE-friendly models with a user interface that is equivalent to their the scikit-learn and XGBoost counterparts
- **Customs models**: Concrete ML supports models that can use quantization-aware training. These are developed by the user using PyTorch or keras/tensorflow and are imported into Concrete ML through ONNX

*Learn more about Concrete ML features in the [documentation](https://docs.zama.ai/concrete-ml).*
<br></br>

### Use cases

By leveraging FHE, Concrete ML can unlock a myriad of new use cases for machine learning, such as enabling secure and private data collaboration, protecting sensitive data while still allowing for analysis, and facilitating machine learning on data-sets that are subject to strict data privacy regulations, for instance

- **Healthcare data analysis**: Improve patient care while maintaining privacy by allowing secure, confidential data sharing between healthcare providers.
- **Financial services**: Facilitate secure financial data analysis for risk management and fraud detection, keeping client information encrypted and safe.
- **Ad campaign tracking**: Create targeted advertising and campaign insights in a post-cookie era, ensuring user privacy through encrypted data analysis.
- **Industries:** Enable predictive maintenance in the cloud while keeping sensitive data confidential, enhancing efficiency and data security.
- **Biometrics:** Give the ability to create user authentication applications without having to reveal their identities.
- **Government:** Enable governments to create digitized versions of their services without having to trust cloud providers.

*See more use cases in the list of [demos](#demos).*
<br></br>

## Table of Contents

- **[Getting Started](#getting-started)**
  - [Installation](#installation)
  - [A simple example](#a-simple-example)
- **[Resources](#resources)**
  - [Demos](#demos)
  - [Tutorials](#tutorials)
  - [Documentation](#documentation)
- **[Working with Concrete ML](#working-with-concrete-ml)**
  - [Citations](#citations)
  - [Contributing](#contributing)
  - [License](#license)
- **[Support](#support)**
  <br></br>

## Getting Started

### Installation

Depending on your OS, Concrete ML may be installed with Docker or with pip:

|                 OS / HW                 | Available on Docker | Available on pip |
| :-------------------------------------: | :-----------------: | :--------------: |
|                  Linux                  |         Yes         |       Yes        |
|                 Windows                 |         Yes         |        No        |
|       Windows Subsystem for Linux       |         Yes         |       Yes        |
|            macOS 11+ (Intel)            |         Yes         |       Yes        |
| macOS 11+ (Apple Silicon: M1, M2, etc.) |     Coming soon     |       Yes        |

Note: Concrete ML only supports Python `3.8`, `3.9` and `3.10`.
Concrete ML can be installed on Kaggle ([see this question on the community for more details](https://community.zama.ai/t/how-do-we-use-concrete-ml-on-kaggle/332)) and on Google Colab.

#### Docker

To install with Docker, pull the `concrete-ml` image as follows:
`docker pull zamafhe/concrete-ml:latest`

#### Pip

To install Concrete ML from PyPi, run the following:

```
pip install -U pip wheel setuptools
pip install concrete-ml
```

*Find more detailed installation instructions in [this part of the documentation](https://docs.zama.ai/concrete-ml/getting-started/pip_installing)*

<p align="right">
  <a href="#about" > ↑ Back to top </a> 
</p>

### A simple example

Here is a simple example which is very close to scikit-learn for a logistic regression :

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

<br></br>
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

*This example is explained in more detail in the [linear model documentation](https://docs.zama.ai/concrete-ml/built-in-models/linear).*

Concrete ML built-in models have APIs that are almost identical to their scikit-learn counterparts. It is also possible to convert PyTorch networks to FHE with the Concrete ML conversion APIs. Please refer to the [linear models](docs/built-in-models/linear.md), [tree-based models](docs/built-in-models/tree.md) and [neural networks](docs/built-in-models/neural-networks.md) documentation for more examples, showing the scikit-learn-like API of the built-in models.

<p align="right">
  <a href="#about" > ↑ Back to top </a> 
</p>

## Resources

### Demos

#### Live demos on Hugging Face

- [Credit card approval](https://huggingface.co/spaces/zama-fhe/credit_card_approval_prediction):  Predicting credit scoring card approval application in which sensitive data can be shared and analyzed without exposing the actual information to neither the three parties involved, nor the server processing it.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/credit_card_approval_prediction/tree/main)
- [Sentiment analysis with transformers](https://huggingface.co/blog/sentiment-analysis-fhe): predicting if an encrypted tweet / short message is positive, negative or neutral, using FHE.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis/tree/main) and the [blog post](https://huggingface.co/blog/sentiment-analysis-fhe)
- [Health diagnosis](https://huggingface.co/spaces/zama-fhe/encrypted_health_prediction): giving a diagnosis using FHE to preserve the privacy of the patient based on a patient's symptoms, history and other health factors.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/encrypted_health_prediction/tree/main)
- [Encrypted image filtering](https://huggingface.co/spaces/zama-fhe/encrypted_image_filtering) : filtering encrypted images by applying filters such as black-and-white, ridge detection, or your own filter.
  - Check the code [here](https://huggingface.co/spaces/zama-fhe/encrypted_image_filtering/tree/main)

#### Other demos

- [Encrypted Large Language Model](use_case_examples/llm/): converting a user-defined part of a Large Language Model for encrypted text generation. This demo shows the trade-off between quantization and accuracy for text generation and shows how to run the model in FHE.
- [Private inference for federated learned models](use_case_examples/federated_learning/): private training of a Logistic Regression model and then importing the model into Concrete ML and performing encrypted prediction.
- [Titanic](use_case_examples/titanic/KaggleTitanic.ipynb): solving the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/). Implemented with XGBoost from Concrete ML, this example comes as a companion of the [Kaggle notebook](https://www.kaggle.com/code/concretemlteam/titanic-with-privacy-preserving-machine-learning), and was the subject of a blogpost in [KDnuggets](https://www.kdnuggets.com/2022/08/machine-learning-encrypted-data.html).
- [CIFAR10 FHE-friendly model with Brevitas](use_case_examples/cifar/cifar_brevitas_training): training a VGG9 FHE-compatible neural network using Brevitas, and a script to run the neural network in FHE. Execution in FHE takes ~4 minutes per image and shows an accuracy of 88.7%.
- [CIFAR10 / CIFAR100 FHE-friendly models with Transfer Learning approach](use_case_examples/cifar/cifar_brevitas_finetuning): series of three notebooks, that convert a pre-trained FP32 VGG11 neural network into a quantized model using Brevitas. The model is fine-tuned on the CIFAR data-sets, converted for FHE execution with Concrete ML and evaluated using FHE simulation. For CIFAR10 and CIFAR100, respectively, our simulations show an accuracy of 90.2% and 68.2%.

*If you have built awesome projects using Concrete ML, please let us know and we will be happy to showcase them here!*
<br></br>

### Tutorials

- [\[Video tutorial\] Train a linear classifier on encrypted data using Concrete ML and Fully Homomorphic Encryption (FHE)](https://www.youtube.com/watch?v=QVsZ33jBlq4)
- [\[Video tutorial\] How To Convert a Scikit-learn Model Into Its Homomorphic Equivalent](https://www.zama.ai/post/how-to-convert-a-scikit-learn-model-into-its-homomorphic-equivalent)
- [Linear Regression Over Encrypted Data With Homomorphic Encryption](https://www.zama.ai/post/linear-regression-using-linear-svr-and-concrete-ml-homomorphic-encryption)
- [How to Deploy a Machine Learning Model With Concrete ML](https://www.zama.ai/post/how-to-deploy-machine-learning-models-with-concrete-ml)
- More [Built-in models tutorials](docs/tutorials/ml_examples.md) and [Deep learning tutorials](docs/tutorials/dl_examples.md)

*Explore more useful resources in [Awesome Zama repo](https://github.com/zama-ai/awesome-zama)*
<br></br>

### Documentation

Full, comprehensive documentation is available here: [https://docs.zama.ai/concrete-ml](https://docs.zama.ai/concrete-ml).

<p align="right">
  <a href="#about" > ↑ Back to top </a> 
</p>

## Working with Concrete ML

### Citations

To cite Concrete ML in academic papers, please use the following entry:

```text
@Misc{ConcreteML,
  title={Concrete {ML}: a Privacy-Preserving Machine Learning Library using Fully Homomorphic Encryption for Data Scientists},
  author={Zama},
  year={2022},
  note={\url{https://github.com/zama-ai/concrete-ml}},
}
```

### Contributing

To contribute to Concrete ML, please refer to [this section of the documentation](docs/developer/contributing.md).
<br></br>

### License

This software is distributed under the **BSD-3-Clause-Clear** license. If you have any questions, please contact us at hello@zama.ai.

<p align="right">
  <a href="#about" > ↑ Back to top </a> 
</p>

## Support

<a target="_blank" href="https://community.zama.ai">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/zama-ai/concrete-ml/assets/157474013/86502167-4ea4-49e9-a881-0cf97d141818">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/zama-ai/concrete-ml/assets/157474013/3dcf41e2-1c00-471b-be53-2c804879b8cb">
  <img alt="Support">
</picture>
</a>

🌟 If you find this project helpful or interesting, please consider giving it a star on GitHub! Your support helps to grow the community and motivates further development.

<p align="right">
  <a href="#about" > ↑ Back to top </a> 
</p>
