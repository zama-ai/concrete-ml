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

The multi-head attention (MHA) and single-head variants now use a rounding approach, significantly improving their execution times in Fully Homomorphic Encryption (FHE) mode.

See [rounded table lookup](https://docs.zama.ai/concrete/v/main-1/tutorials/rounded_table_lookups) from the Concrete library for more details.

For the single-head model, the execution time is 166.38 seconds on a single 196 cores CPU machine (an hp7c from AWS). The multi-head attention model, which is a full attention block from GPT-2, now runs in about 862.97 seconds (~14 minutes) under the same conditions. All these timings are actual FHE execution on encrypted data.

Note that, computations were done using 8 input tokens.

You can replicate these results by running the [QGPT2Evaluate.ipynb](./QGPT2Evaluate.ipynb) notebook.

## Additional Classes and Functions

The code includes additional classes and functions that are used for quantization and other operations. These classes and functions include:

- `Quantizer`: Implements the quantization and de-quantization of arrays.
- `DualArray`: Represents an array in both the floating point and integer domains.
- `QGPT2`: Base class that implements the quantized operators needed for the quantized GPT-2 models.
- `QGPT2LMHeadModel`: Base class for quantized GPT-2 models.
- `SingleHeadQGPT2Model`: Implementation of Projections + Single Attention Head quantized GPT-2 model.
- `MultiHeadsQGPT2Model`: Implementation of Multi-Head Attention quantized GPT-2 model.

These classes and functions are used internally in the code and do not need to be directly called by the user.
