# What is **Concrete ML**?

## Introduction

**Concrete-ML** is an open-source package, built on top of **Concrete-Numpy**. Its purpose is to allow data scientists without any prior knowledge of cryptography to automatically turn machine learning (ML) models into their fully homomorphic encryption (FHE) equivalent.

FHE is a powerful cryptographic tool that allows cloud service providers to perform computations directly on encrypted data without needing to decrypt the data first. With FHE, you can build services which ensure full user privacy and are the complete equivalent of their insecure counterparts.

FHE is also a killer feature regarding data breaches, as anything done on the server is done over encrypted data. Even if the server is compromised, there is no leak of useful data.

A major goal of our first **Concrete-ML** release is to make the adoption of **Concrete-ML** as simple as possible for users of popular machine learning frameworks. This is why we try to provide APIs, which are as close as possible to what data scientists are already using.

## Philosophy of the Design

Our primary concern in this release was the ease of adoption of our framework. That is why we built APIs, which should feel natural to data scientists. While performance is also an important concern for deployment of FHE machine learning models, improvements on this front will come in future releases.

To this end, we have decided to mimic the APIs of scikit-learn and XGBoost for machine learning models (linear models and tree-based models) and of torch for deep learning models. We refer readers to [linear models](linear.md), [tree-based models](tree.md) and [neural networks](neural_network.md) documentations, which show how similar our APIs are to their non-FHE counterparts.

## Concrete Stack

**Concrete-ML** is built on top of Zama’s **Concrete** stack. It uses [**Concrete-Numpy**](https://github.com/zama-ai/concrete-numpy), which itself uses the [**Concrete-Compiler**](https://pypi.org/project/concrete-compiler/).

The **Concrete-Compiler** takes MLIR code as input representing a computation circuit and compiles it to an executable using **Concrete** primitives to perform the computations.

We refer the reader to **Concrete-Numpy** [documentation](https://docs.zama.ai/concrete-numpy/stable/) and, more generally, to the documentation of the whole **Concrete-Framework** for [more information](https://docs.zama.ai).

## A work in progress

This is the very first version of the package, so, even if we are able to show very appealing examples (see [our examples](advanced_examples/index.rst), it is neither complete nor bug-free and not quite as efficient as one would hope. We will improve it in further releases.

The main difficulty is that some models currently do not work well due to the fact that **Concrete-Library** only supports 8 bits. Because of this, we sometimes have to quantize too much, which has a strong negative impact on certain models. Further, because **Concrete-Compiler** is also a work in progress, we have FHE programs which are sometimes too slow (notably, parallelism is lacking and will be updated soon) or they may require a massive amount of RAM. Last but not least, we have selected some models for this release, and we will add more in future releases.

## Resources

The interested reader has even more resources to review, in addition to this documentation:

1. Our [community page](https://community.zama.ai/c/concrete-ml), the link for which can be found at the top right of doc pages.
1. The varied [blogs](https://www.zama.ai/blog) we publish. Notably, [this blog post](https://www.zama.ai/post/quantization-of-neural-networks-for-fully-homomorphic-encryption) describes the use of a Poisson regressor to tackle a real-life use case in a privacy-preserving setting.

Additionally, we plan to publish academic and white papers explaining interesting aspects of our work, covering both the engineering and scientific sides of our offering.
