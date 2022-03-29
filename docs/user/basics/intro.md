# What is **Concrete ML**?

## Introduction

**Concrete ML** is an open-source package, built on top of **Concrete Numpy**. Its purpose is to allow data scientists without any prior knowledge of cryptography to automatically turn machine learning (ML) models into their fully homomorphic encryption (FHE) equivalent.

FHE is a powerful cryptographic tool, which allows cloud service providers to perform computations directly on encrypted data without needing to decrypt first. With FHE, you can build services which ensure full privacy of the user and are the complete equivalent of their insecure counterparts.

FHE is also a killer feature regarding data breaches: as anything done on the server is done over encrypted data, even if the server is compromised, there is no leak of useful data.

A major goal of our first **Concrete ML** release is to make the adoption of **Concrete ML** as simple as it can be for users of popular machine learning frameworks. This is why we try to provide APIs which are as close as possible to what data scientists are already using.

## Organization of the documentation

We have divided our documentation into several parts:

- basic elements, notably a description of the installation, that you are currently reading
- for users of **Concrete ML**: tutorials, how-tos and deeper explanations
- an API guide of the different functions of **Concrete ML**, directly done by parsing its source code
- and, finally, a developer section, for both internal or external contributors to **Concrete ML**.

## A work in progress

This is the very first version of the package, and, even if we are able to show very appealing examples (see [our examples](../../user/advanced_examples/index.md)), it is neither complete nor bug-free and is not as efficient as one would hope. We will improve it in further releases.

The main difficulty is that currently some models do not work well due to the fact that **Concrete Library** only supports 7 bits. Because of this, we sometimes have to quantize too much, which has a strong negative impact on certain models. Also, because **Concrete Compiler** is also a work in progress, we have FHE programs which are sometimes too slow (notably, more parallelism will be added soon) or may require a massive amount of RAM. Last but not least, we have selected some models for this release, and we will add more of them in next releases.
