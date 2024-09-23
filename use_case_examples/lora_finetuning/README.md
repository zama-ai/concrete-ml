# Privacy Preserving GPT2 LoRA

This project demonstrates how to fine-tune GPT-2 using Low-Rank Adaptation (LoRA) weights with Fully Homomorphic Encryption (FHE). The goal is to train a specialized model in a privacy-preserving manner, with minimal memory requirements.

## Overview

Fine-tuning large language models typically requires access to sensitive data, which can raise privacy concerns. By leveraging FHE, we can perform computations on encrypted data, ensuring that the data remains private throughout the training process. In this approach, the LoRA weights are only known to the user who owns the data and the memory hungry foundation model remains on the server.

## Key Features

- **LoRA Fine-Tuning**: Fine-tune GPT-2 by adapting low-rank weights.
- **Hybrid Model**: Combine traditional and encrypted computations for optimal performance.
- **Low Memory Requirements**: Minimal client-side memory needed for LoRA weights.

## Setup

### Installation

Install the required packages:

<!--pytest-codeblocks:skip-->

```sh
pip install -r requirements.txt
```

## Usage

### Prepare the Dataset

Replace the data-set in the `data_finetune` directory to the one you want to use for fine-tuning.

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

The fine-tuned model can generate specialized text based on the provided data-set while ensuring data privacy through FHE.

After fine-tuning, the model's weights are distributed between the client and server as follows:

- Total weights removed from the server: 68.24%
- LoRA weights kept on the client: 147,456 (approximately 0.12% of the original model's weights)

Note that the embedding are not considered for now but contain a significant amount of weights (around 30%) for GPT2. They will be considered in a future version of Concrete ML.

## Conclusion

This project showcases the potential of combining LoRA and FHE to fine-tune language models in a privacy-preserving manner. By following the steps outlined in the notebook, you can adapt this approach to your own data-sets and use cases.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT](https://github.com/huggingface/peft)
- [Concrete ML](https://github.com/zama-ai/concrete-ml)
