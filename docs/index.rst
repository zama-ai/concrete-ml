
What is **Concrete ML**?
==================================

Introduction
----------------------------------

**Concrete-ML** is an open-source package, whose purpose is to allow data scientists without any prior knowledge of cryptography to automatically turn machine learning (ML) models into their fully homomorphic encryption (FHE) equivalent.

FHE is a powerful cryptographic tool that allows cloud service providers to perform computations directly on encrypted data without needing to decrypt the data first. With FHE, you can build services which ensure full user privacy and are the complete equivalent of their insecure counterparts. FHE is also a killer feature regarding data breaches, as anything done on the server is done over encrypted data. Even if the server is compromised, there is no leak of sensitive data.

An easy to use tool for data scientists
-----------------------------------------

**Concrete-ML** aims to facilitate adoption of privacy preserving ML for users of popular machine learning frameworks by providing APIs which are as close as possible to what data scientists are already using. **Concrete-ML** mimics the APIs of scikit-learn and XGBoost for machine learning models (linear models and tree-based models) and of torch for deep learning models. We refer readers to `linear models <linear.md>`__, `tree-based models <tree.md>`__ and `neural networks <quantized_neural_networks.md>`__ documentations, which show how similar APIs are to their non-FHE counterparts.

Concrete Stack
----------------------------------

**Concrete-ML** is built on top of Zama’s **Concrete** stack. It uses `Concrete-Numpy <https://github.com/zama-ai/concrete-numpy>`__, which itself uses the `Concrete-Compiler <https://pypi.org/project/concrete-compiler>`__, which is based on the `Concrete-Library <https://docs.zama.ai/concrete/core-lib/main/>`__. We refer the reader to **Concrete-Numpy** `documentation <https://docs.zama.ai/concrete-numpy/stable/>`__ and, more generally, to the documentation of the whole **Concrete-Framework** for `more information <https://docs.zama.ai>`__ .

A work in progress
----------------------------------

One of the main current difficulties is that some models currently do not work well due to the fact that **Concrete-Library** only supports 8 bits. Because of this, we sometimes have to quantize too much, which has a strong negative impact on certain models. Further, because **Concrete-Compiler** is also a work in progress, we have FHE programs which are sometimes too slow or they may require a massive amount of RAM: improvements on this front will come in future releases.

Nevertheless, these restrictions, typical of a work-in-progress tool on a very recent topic such as FHE, do not prevent us to show very appealing examples (see `ML examples <ml_examples.md>`__ and `DL examples <dl_examples.md>`__).

Resources
----------------------------------

The interested reader has even more resources to review, in addition to this documentation:

#. The `community page <https://community.zama.ai/c/concrete-ml>`__.
#. The varied `blogs <https://www.zama.ai/blog>`__ we publish.

Additionally, academic and white papers will be published, explaining interesting aspects of this work, covering both the engineering and scientific sides.

.. toctree::
    :maxdepth: 0
    :hidden:

    self

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Introduction

    pip_installing.md
    docker_installing.md
    simple_compilation.md
    client_server.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Concrete-ML Model library

    linear.md
    tree.md
    quantized_neural_networks.md
    ml_examples.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Machine learning in FHE

    fhe_constraints.md
    quantization.md
    pruning.md
    compilation.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Deep learning in FHE

    torch_support.md
    onnx_support.md
    fhe_assistant.md
    fhe_friendly_models.md
    dl_examples.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Pre post processing

    concrete_numpy.md
    pandas.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Developper Guide

    project_setup.md
    docker_setup.md
    documenting.md
    debug_support_submit_issues.md
    releasing.md
    contributing.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: Architecture

    onnx_pipeline.md
    quantized_ops.md
    hummingbird_usage.md
    skorch_usage.md

.. toctree::
    :maxdepth: 0
    :hidden:
    :caption: API

    API <_apidoc/modules.rst>
