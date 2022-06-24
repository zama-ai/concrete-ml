# What is **Concrete ML**?

## Introduction

**Concrete-ML** is an open-source package, whose purpose is to allow data scientists without any prior knowledge of cryptography to automatically turn machine learning (ML) models into their fully homomorphic encryption (FHE) equivalent.

FHE is a powerful cryptographic tool that allows cloud service providers to perform computations directly on encrypted data without needing to decrypt the data first. With FHE, you can build services which ensure full user privacy and are the complete equivalent of their insecure counterparts. FHE is also a killer feature regarding data breaches, as anything done on the server is done over encrypted data. Even if the server is compromised, there is no leak of sensitive data.

## An easy to use tool for data scientists

**Concrete-ML** aims to facilitate adoption of privacy preserving ML for users of popular machine learning frameworks by providing APIs which are as close as possible to what data scientists are already using. **Concrete-ML** mimics the APIs of scikit-learn and XGBoost for machine learning models (linear models and tree-based models) and of torch for deep learning models. We refer readers to [linear models](linear.md%3E), [tree-based models](tree.md%3E) and [neural networks](quantized_neural_networks.md) documentations, which show how similar APIs are to their non-FHE counterparts.

## Concrete Stack

**Concrete-ML** is built on top of Zama’s **Concrete** stack. It uses [Concrete-Numpy](https://github.com/zama-ai/concrete-numpy), which itself uses the [Concrete-Compiler](https://pypi.org/project/concrete-compiler), which is based on the [Concrete-Library](https://docs.zama.ai/concrete/core-lib/main/). We refer the reader to **Concrete-Numpy** [documentation](https://docs.zama.ai/concrete-numpy/stable/) and, more generally, to the documentation of the whole **Concrete-Framework** for [more information](https://docs.zama.ai).

## A work in progress

One of the main current difficulties is that some models currently do not work well due to the fact that **Concrete-Library** only supports 8 bits. Because of this, we sometimes have to quantize too much, which has a strong negative impact on certain models. Further, because **Concrete-Compiler** is also a work in progress, we have FHE programs which are sometimes too slow or they may require a massive amount of RAM: improvements on this front will come in future releases.

Nevertheless, these restrictions, typical of a work-in-progress tool on a very recent topic such as FHE, do not prevent us to show very appealing examples (see [ML examples](ml_examples.md) and [ML examples](dl_examples.md)).

## Resources

The interested reader has even more resources to review, in addition to this documentation:

- The [community page](https://community.zama.ai/c/concrete-ml).
- The varied [blogs](https://www.zama.ai/blog) we publish.

Additionally, academic and white papers will be published, explaining interesting aspects of this work, covering both the engineering and scientific sides.
