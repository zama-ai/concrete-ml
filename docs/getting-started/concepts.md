# Key concepts

This document explains the essential cryptographic terms and the important concepts of Concrete ML model lifecycle with Fully Homomorphic Encryption (FHE).

Concrete ML is built on top of Concrete, which enables the conversion from NumPy programs into FHE circuits.

## Table of Contents

1. [Lifecycle of a Concrete ML model](#lifecycle-of-a-concrete-ml-model)
1. [Cryptography concepts](#cryptography-concepts)
1. [Ciphertext formats](#ciphertext-formats)
1. [Model accuracy considerations under FHE constraints](#model-accuracy-considerations-under-fhe-constraints)

## Lifecycle of a Concrete ML model

With Concrete ML, you can train a model on clear or encrypted data, then deploy it to predict on encrypted inputs. During deployment, data can be pre-processed while being encrypted. Therefore, data stay encrypted during the entire lifecycle of the machine learning model, with some limitations.

### I. Model development

1. **Training:** A model is trained either using plaintext (non-encrypted) training data, or encrypted training data.

1. **Quantization:** Quantization converts inputs, model weights, and all intermediate values of the inference computation to integer equivalents. More information is available [here](../explanations/quantization.md). Concrete ML performs this step in two ways depending on model type:

   - During training (Quantization Aware Training): by adding quantization layers in the neural network model, weights can be forced to have discrete values and activation quantization parameters are optimized through gradient descent. QAT requires re-training a neural network with these quantization layers.
   - After training (Post Training Quantization): the floating point neural network is kept as-is and a calibration step determines quantization parameters for each layer. No re-training is necessary and thus, no training data or labels are needed to convert a neural network to FHE using PTQ.

1. **Simulation:** Simulation allows you to execute a model that was quantized, to measure its accuracy in FHE, and to determine the modifications required to make it FHE compatible. Simulation is described in more detail [here](../explanations/compilation.md#fhe-simulation).

1. **Compilation:** After quantizing the model and confirming that it has good FHE accuracy through simulation, the model then needs to be compiled using Concrete's FHE Compiler to produce an equivalent FHE circuit. This circuit is represented as an MLIR program consisting of low level cryptographic operations. You can read more about FHE compilation [here](../explanations/compilation.md), MLIR [here](https://mlir.llvm.org/), and about the low-level Concrete library [here](https://github.com/zama-ai/concrete).

1. **Inference:** The compiled model can then be executed on encrypted data, once the proper keys have been generated. The model can also be deployed to a server and used to run private inference on encrypted inputs.

You can find examples of the model development workflow [here](../tutorials/ml_examples.md).

### II. Model deployment

1. **Pre-processing:** Data owners(client) can generate keys to encrypt/decrypt data and store it in a [DataFrame](../built-in-models/encrypted_dataframe.md) for further processing on a server. The server can pre-process such data with pre-compiled circuits, to prepare it for encrypted training or inference.

1. **Client/server model deployment:** In a client/server setting, Concrete ML models can be exported to:

   - Allow the client to generate keys, encrypt, and decrypt.
   - Provide a compiled model that can run on the server to perform inference on encrypted data.

1. **Key generation:** The data owner (client) needs to generate a set of keys:

   - A private encryption key to encrypt/decrypt their data and results
   - A public evaluation key for the model's FHE evaluation on the server.

1. **Ciphertext formats** The server-side application can be configured to accept different types of ciphertexts from the client, depending on the type of application. See [Ciphertext formats](#ciphertext-formats) for more details.

You can find an example of the model deployment workflow [here](../advanced_examples/ClientServer.ipynb).

## Cryptography concepts

Concrete ML and Concrete abstract the details of the underlying cryptography scheme, TFHE. However, understanding some cryptography concepts is still useful:

- **Encryption and decryption:** Encryption converts human-readable information (plaintext) into data (ciphertext) that is unreadable by a human or computer unless with the proper key. Encryption takes plaintext and an encryption key and produces ciphertext, while decryption is the reverse operation.

- **Encrypted inference:** FHE allows third parties to execute a machine learning model on encrypted data. The inference result is also encrypted and can only be decrypted by the key holder.

- **Key generation:** Cryptographic keys are generated using random number generators. Key generation can be time-consuming and produce large keys, but each model used by a client only requires key generation once.

  - **Private encryption key**: A private encryption key is a series of bits used within an encryption algorithm for encrypting data so that the corresponding ciphertext appears random.

  - **Public evaluation key**: A public evaluation key is used to perform homomorphic operations on encrypted data, typically by a server.

- **Guaranteed correctness of encrypted computations:** To ensure security, TFHE adds random noise to ciphertexts. Depending on the noise parameters, it can cause errors during encrypted data processing. By default, Concrete ML uses parameters that guarantee the correctness of encrypted computations, so the results on encrypted data equals to those of simulations on clear data.

- **Programmable Boostrapping (PBS)** : Programmable Bootstrapping enables the homomorphic evaluation of any function of a ciphertext, with a controlled level of noise. Learn more about PBS in [this paper](https://eprint.iacr.org/2021/091).

- **Ciphertext formats**: To represent encrypted values, Concrete ML offers two options: the default Concrete ciphertext format, which is supported by all ML models and highly optimized for performance, or the block-based TFHE-rs radix format, which supports larger values, is forward-compatible, and suitable for Blockchain applications, but is limited to certain types of ML models.

For a deeper understanding of the cryptography behind the Concrete stack, refer to the [whitepaper on TFHE and Programmable Boostrapping](https://whitepaper.zama.ai/) or [this series of blogs](https://www.zama.ai/post/tfhe-deep-dive-part-1).

## Ciphertext formats

Two different types of ciphertexts are usable by Concrete ML for model input/outputs.

1. _Concrete_ LWE ciphertexts (default):

   By default, Concrete ML uses Concrete LWE ciphertexts with crypto-system parameters that are tailored to each ML model. These parameters may vary between different versions of Concrete ML. Thus, the encryption crypto-parameters may change at any point. Some implications are:

- Typically, a server-side application provides the client with its encryption cryptographic parameters.
- When the application is updated, the client downloads the new cryptographic parameters.
- Ciphertexts encrypted with a set of cryptographic parameters can not be re-used for a model compiled with different cryptographic parameters

2. _TFHE-rs radix_ ciphertexts:

   Concrete ML also supports \_TFHE-rs radix _ ciphertexts, which rely on a universal and forward-compatible parameter set. Therefore:
   In this setting, a conversion layer is added to the ML model, potentially resulting in a 4–5× latency overhead.

- Ciphertexts encrypted with the universal cryptographic parameters can be used at any point in the future with any ML model.
- In this setting, a conversion layer is added to the ML model. This conversion may imply a 4-5x slowdown for model latency.

## Model accuracy considerations under FHE constraints

FHE requires all inputs, constants, and intermediate values to be integers of maximum 16 bits. To make machine learning models compatible with FHE, Concrete ML implements some techniques with accuracy considerations:

- **Quantization**: Concrete ML quantizes inputs, outputs, weights, and activations to meet FHE limitations. See [the quantization documentation](../explanations/quantization.md) for details.

  - **Accuracy trade-off**: Quantization may reduce accuracy, but careful selection of quantization parameters or of the training approach can mitigate this. Concrete ML offers built-in quantized models; users only configure parameters like bit-width. For more details of quantization configurations, see [the advanced quantization guide](../explanations/quantization.md#configuring-model-quantization-parameters).

- **Additional methods**: Dimensionality reduction and pruning are additional ways to make programs compatible for FHE. See [Poisson regression example](../advanced_examples/PoissonRegression.ipynb) for dimensionality reduction and [built-in neural networks](../built-in-models/neural-networks.md) for pruning.
