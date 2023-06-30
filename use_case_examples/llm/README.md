# Secure Large Language Models Using Fully Homomorphic Encryption (FHE)

Large Language Models (LLMs), like GPT-2, have been game-changers in diverse fields including programming, content creation, text analysis, web search, and remote learning. However, the privacy of user queries processed by these LLMs has been a cause for concern.

This project introduces a unique solution to this issue using Fully Homomorphic Encryption (FHE), a process that allows computations on encrypted data. This empowers LLMs to operate on this encrypted data without any compromise on prediction accuracy or the risk of Intellectual Property (IP) theft.

## Overview

This repository contains Python code illustrating how to apply GPT-2, a Large Language Model, to encrypted data. It includes different variants to have GPT-2 models run on encrypted data.

## Installation

You will need Python 3 and the following dependencies to use the code from this repository:

- concrete-ml
- transformers

You can install these dependencies with the following command:

<!--pytest-codeblocks:skip-->

```bash
pip install -r requirements.txt
```

## How to Use

How to Use
You can use this code to predict the next token based on an input sentence via various GPT-2 models. There are three distinct modes of inference:

1. Clear Quantized: Inference on unencrypted data.
1. FHE Simulation: Simulates the inference process using FHE.
1. FHE Execution: Integrates all FHE stages such as key generation, encryption, FHE inference, and decryption.

The execution mode can easily be switched between the three options we just described.
Case 1.:

<!--pytest-codeblocks:skip-->

```python
proj_single_head_qgpt2.set_fhe_mode(fhe="disable")
```

Case 2.:

<!--pytest-codeblocks:skip-->

```python
proj_single_head_qgpt2.set_fhe_mode(fhe="simulate")
```

Case 3.:

<!--pytest-codeblocks:skip-->

```python
proj_single_head_qgpt2.set_fhe_mode(fhe="execute")
```

The notebook [QGPT2Evaluate.ipynb](./QGPT2Evaluate.ipynb) provides the implementation for all these inference modes.

## Model Variations

This project demonstrates that any part of your model can be encrypted. We offer two methods for operating LLMs on encrypted data:

- Single Attention Head (single head): This variant works with a single attention head on encrypted data.
- Multi-Head Attention (MHA): This variant employs 12 attention heads or an entire block/layer of GPT-2 on encrypted data.

## Evaluation

The repository also comprises an evaluation of the accuracy of the aforementioned variants. This assessment evaluates the top-k accuracy for each logits of each input token against those from the original GPT-2 model. The assessment is conducted for various numbers of quantization bit-width and different top-k values.

The evaluation details are also available in the [QGPT2Evaluate.ipynb](./QGPT2Evaluate.ipynb) notebook.

## FHE Execution

The multi-head attention (MHA) and single-head variants, in their current implementation, have not been optimized sufficiently in terms of precision to provide relevant execution times.

If you aim to operate in FHE, it would be necessary to reduce the overall precision. A potential way to effectively achieve this is by using [rounded table lookup](https://docs.zama.ai/concrete/v/main-1/tutorials/rounded_table_lookups) from the Concrete library.

Also note that, as with the standard transformer architecture, the execution time is heavily reliant on the number of input tokens. For example, a precision of 4 bits for the single head model takes 11 minutes to run on a single 128 cores CPU machine (an m6i.metal from aws). You can easily replicate this run by setting the FHE mode to "execute" as shown below:

<!--pytest-codeblocks:skip-->

```python
proj_single_head_qgpt2.set_fhe_mode(fhe="execute")
```

and then doing a standard inference on an input.

## Additional Classes and Functions

The code includes additional classes and functions that are used for quantization and other operations. These classes and functions include:

- `Quantizer`: Implements the quantization and de-quantization of arrays.
- `DualArray`: Represents an array in both the floating point and integer domains.
- `QGPT2`: Base class that implements the quantized operators needed for the quantized GPT-2 models.
- `QGPT2LMHeadModel`: Base class for quantized GPT-2 models.
- `SingleHeadQGPT2Model`: Implementation of Projections + Single Attention Head quantized GPT-2 model.
- `MultiHeadsQGPT2Model`: Implementation of Multi-Head Attention quantized GPT-2 model.

These classes and functions are used internally in the code and do not need to be directly called by the user.
