# Key concepts

Concrete ML is built on top of Concrete, which enables NumPy programs to be converted into FHE circuits.

## Lifecycle of a Concrete ML model

Concrete ML models can be trained on clear or encrypted data, then deployed to predict on encrypted inputs. During deployment
data pre-processing can be done on encrypted data. Therefore, data can be encrypted during the entire lifecycle of the machine
learning model, with some limitations.

### I. Model development

1. **training:** A model is trained either using plaintext, non-encrypted, training data, or encrypted training data.
1. **quantization:** The model is converted into an integer equivalent using quantization. Concrete ML performs this step either during training (Quantization Aware Training) or after training (Post-training Quantization), depending on model type. Quantization converts inputs, model weights, and all intermediate values of the inference computation to integers. More information is available [here](../explanations/quantization.md).
1. **simulation:** Testing FHE models on very large data-sets can take a long time. Furthermore, not all models are compatible with FHE constraints out of the box. Simulation allows you to execute a model that was quantized, to measure the accuracy it would have in FHE, but also to determine the modifications required to make it FHE compatible. Simulation is described in more detail [here](../explanations/compilation.md#fhe-simulation).
1. **compilation:** Once the model is quantized, simulation can confirm it has good accuracy in FHE. The model then needs to be compiled using Concrete's FHE Compiler to produce an equivalent FHE circuit. This circuit is represented as an MLIR program consisting of low level cryptographic operations. You can read more about FHE compilation [here](../explanations/compilation.md), MLIR [here](https://mlir.llvm.org/), and about the low-level Concrete library [here](https://github.com/zama-ai/concrete).
1. **inference:** The compiled model can then be executed on encrypted data, once the proper keys have been generated. The model can also be deployed to a server and used to run private inference on encrypted inputs.

You can find examples of the model development workflow [here](../tutorials/ml_examples.md).

### II. Model deployment

1. **pre-processing:** Data owners can encrypt data and store it in a [data-frame](../built-in-models/encrypted_dataframe.md) for
   further processing on a server. The server can pre-process such data, to prepare it for encrypted training or inference. Clients
   generate keys, encrypt and decrypt data-frames, while the Concrete ML-enabled server has pre-compiled circuits
   that perform pre-processing.
1. **client/server model deployment:** In a client/server setting, Concrete ML models can be exported in a way that:
   - allows the client to generate keys, encrypt, and decrypt.
   - provides a compiled model that can run on the server to perform inference on encrypted data.
1. **key generation:** The data owner (client) needs to generate a set of keys: a private key (to encrypt/decrypt their data and results) and a public evaluation key (for the model's FHE evaluation on the server).

You can find an example of the model deployment workflow [here](../advanced_examples/ClientServer.ipynb).

## Cryptography concepts

Concrete ML and Concrete are tools that hide away the details of the underlying cryptography scheme, called TFHE. However, some cryptography concepts are still useful when using these two toolkits:

1. **encryption/decryption:** These operations transform plaintext (i.e., human-readable information) into ciphertext (i.e., data that contains a form of the original plaintext that is unreadable by a human or computer without the proper key to decrypt it). Encryption takes plaintext and an encryption key and produces ciphertext, while decryption is the inverse operation.
1. **encrypted inference:** FHE allows a third party to execute (i.e., run inference or predict) a machine learning model on encrypted data (a ciphertext). The result of the inference is also encrypted and can only be read by the person who receives the decryption key.
1. **key generation:** Cryptographic keys need to be generated using random number generators. Their size may be large and key generation may take a long time. However, keys only need to be generated once for each model used by a client.
1. **private key:** A private key is a series of bits used within an encryption algorithm for encrypting data so that the corresponding ciphertext appears random.
1. **public evaluation key:** A public evaluation key is used to perform homomorphic operations on encrypted data, typically by a server.
1. **guaranteed correctness of encrypted computations:** To achieve security, TFHE, the underlying encryption scheme, adds random noise to ciphertexts. This can induce errors during processing of encrypted data, depending on noise parameters. By default, Concrete ML uses parameters that ensure the correctness of the encrypted computation, so there is no need to account for noise parametrization. Therefore, the results on encrypted data will be the same as the results of simulation on clear data.

While Concrete ML users only need to understand the cryptography concepts above, for a deeper understanding of the cryptography behind the Concrete stack, please see the [whitepaper on TFHE and Programmable Boostrapping](https://whitepaper.zama.ai/) or [this series of blogs](https://www.zama.ai/post/tfhe-deep-dive-part-1).

## Model accuracy considerations under FHE constraints

To respect FHE constraints, all numerical programs that include non-linear operations over encrypted data must have all inputs, constants, and intermediate values represented with integers of a maximum of 16 bits.

Concrete ML quantizes the input data and model outputs in the same way as weights and activations. The main levers to control accumulator bit-width are the number of bits used for the inputs, weights, and activations of the model. These parameters are crucial to comply with the constraint on accumulator bit-widths. Please refer to [the quantization documentation](../explanations/quantization.md) for more details about how to develop models with quantization in Concrete ML.

These methods may cause a reduction in the accuracy of the model since its representative power is diminished. Carefully choosing a quantization approach can alleviate accuracy loss, all the while allowing compilation to FHE. Concrete ML offers built-in models that include quantization algorithms, and users only need to configure some of their parameters, such as the number of bits, discussed above. See [the advanced quantization guide](../explanations/quantization.md#configuring-model-quantization-parameters) for information about configuring these parameters for various models.

Additional specific methods can help to make models compatible with FHE constraints. For instance, dimensionality reduction can reduce the number of input features and, thus, the maximum accumulator bit-width reached within a circuit. Similarly, sparsity-inducing training methods, such as pruning, deactivate some features during inference. For now, dimensionality reduction is considered as a pre-processing step, while pruning is used in the [built-in neural networks](../built-in-models/neural-networks.md).

The configuration of model quantization parameters is illustrated in the advanced examples for [Linear and Logistic Regressions](../tutorials/ml_examples.md) and dimensionality reduction is shown in the [Poisson regression example](../advanced_examples/PoissonRegression.ipynb).
