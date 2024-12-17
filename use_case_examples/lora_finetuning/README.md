# Privacy Preserving Language Models LoRA Fine-tuning

This use case demonstrates how to fine-tune language models (GPT-2 and LLaMA) using Low-Rank Adaptation (LoRA) weights with Fully Homomorphic Encryption (FHE). The goal is to train specialized models in a privacy-preserving manner, with minimal memory requirements.

## Overview

Fine-tuning large language models typically requires access to sensitive data, which can raise privacy concerns. By leveraging FHE, we can perform computations on encrypted foundation model weights, ensuring that the data remain private throughout the training process. The LoRA weights are kept in clear on the client side.


## Key Features

- **LoRA Fine-Tuning**: Fine-tune language models by adapting low-rank weights
- **Hybrid Model**: Combine encrypted foundation model weights with clear LoRA weights for optimal performance
- **Low Memory Requirements**: Minimal client-side memory needed for LoRA weights
- **Multiple Approaches**: 
  - Custom training implementation for GPT-2
  - Simplified API-based approach for LLaMA using the `LoraTrainer`

## Setup

### Installation

Install the required packages:

<!--pytest-codeblocks:skip-->

```sh
pip install -r requirements.txt
```

## Usage

### Available Notebooks

The repository includes two example notebooks:

1. **GPT2FineTuneHybrid.ipynb**: 
   - Uses a custom training implementation
   - Fine-tunes GPT-2 on a small Q&A data-set about FHE
   - Shows low-level control over the training process

2. **LLamaFineTuning.ipynb**:
   - Uses Concrete ML's `LoraTrainer` API for simplified implementation
   - Fine-tunes LLaMA on Concrete ML code examples
   - Shows how to use the high-level API for encrypted fine-tuning

### Prepare the data-set

Each notebook includes its own data-set:
- GPT-2 uses a small Q&A data-set about FHE in `data_finetune/what_is_fhe.txt`
- LLaMA uses Concrete ML code examples in `data_finetune/data-set.jsonl`

### Run the Fine-Tuning Script

Execute the Jupyter notebook `GPT2FineTuneHybrid.ipynb` to start the fine-tuning process. The notebook is structured into several steps:

## Deployment/Production Scenario

In a deployment or production scenario, the model can be fine-tuned as follows:

1. **Server Setup**: The server hosts a foundation model with generic weights.
1. **Client Setup**: The user (client) has a set of LoRA weights and the sensitive data required for fine-tuning.
1. **Fine-Tuning Process**:
   - The client requests inference and backward passes from the server, which uses the generic weights/parameters.
   - Any computation that requires the LoRA weights is executed on the client's end.
1. **Storage**: The LoRA weights are stored on the client's end for later inference, ensuring full privacy of both the specialized model and the sensitive data.

## Results


### GPT-2 Results
After fine-tuning, the model's weights are distributed between the client and server as follows:

- Total weights removed from the server: 68.24%
- LoRA weights kept on the client: 147,456 (approximately 0.12% of the original model's weights)

Note that the embeddings are not considered for now but contain a significant amount of weights (around 30%) for GPT2. They will be considered in a future version of Concrete ML.

### LLaMA Results

TBD

## Conclusion

This project showcases the potential of combining LoRA and FHE to fine-tune language models in a privacy-preserving manner. By following the steps outlined in the notebook, you can adapt this approach to your own data-sets and use cases.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT](https://github.com/huggingface/peft)
- [Concrete ML](https://github.com/zama-ai/concrete-ml)
