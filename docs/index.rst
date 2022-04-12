What is **Concrete ML**?
==================================

Introduction
----------------------------------

**Concrete-ML** is an open-source package, built on top of **Concrete-Numpy**. Its purpose is to allow data scientists without any prior knowledge of cryptography to automatically turn machine learning (ML) models into their fully homomorphic encryption (FHE) equivalent.

FHE is a powerful cryptographic tool that allows cloud service providers to perform computations directly on encrypted data without needing to decrypt the data first. With FHE, you can build services which ensure full user privacy and are the complete equivalent of their insecure counterparts.

FHE is also a killer feature regarding data breaches, as anything done on the server is done over encrypted data. Even if the server is compromised, there is no leak of useful data.

A major goal of our first **Concrete-ML** release is to make the adoption of **Concrete ML** as simple as possible for users of popular machine learning frameworks. This is why we try to provide APIs, which are as close as possible to what data scientists are already using.

Organization of the documentation
----------------------------------

We have divided our documentation into several parts:

- installation, tutorials, how-tos and deeper explanations, exclusively for users of **Concrete-ML**;
- an API guide of the different functions of **Concrete-ML**, created by directly parsing its source code;
- and, finally, a developer section, for both internal or external contributors to **Concrete-ML**.

A work in progress
----------------------------------

This is the very first version of the package, so, even if we are able to show very appealing examples (see `our examples <user/advanced_examples/index.rst>`__), it is neither complete nor bug-free and not quite as efficient as one would hope. We will improve it in further releases.

The main difficulty is that some models currently do not work well due to the fact that **Concrete-Library** only supports 7 bits. Because of this, we sometimes have to quantize too much, which has a strong negative impact on certain models. Further, because **Concrete-Compiler** is also a work in progress, we have FHE programs which are sometimes too slow (notably, parallelism is lacking and will be updated soon) or they may require a massive amount of RAM. Last but not least, we have selected some models for this release, and we will add more in future releases.

Table of contents
----------------------------------

.. toctree::
    :maxdepth: 2

    User Guide <user/index.rst>
    API <_apidoc/modules.rst>
    Developer <dev/index.rst>
