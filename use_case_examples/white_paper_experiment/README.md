# Reproducing Paper **Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks** : Benchmarking Results

This notebook replicates experiments from the paper [_Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks_](https://whitepaper.zama.ai/), published in 2021. 
It provides an in-depth analysis of the deep neural network architectures NN-20 and NN-50, along with their training processes using floating point precision and their [quantization](https://docs.zama.ai/concrete-ml/explanations/quantization) using the Quantization Aware Training (QAT) and Post Training Quantization (PTQ) methods. 


## Installation

- First, create a virtual env and activate it:

<!--pytest-codeblocks:skip-->

```bash
python3.8 -m venv .venv
source .venv/bin/activate
```

- Then, install required packages:

<!--pytest-codeblocks:skip-->

```bash
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```
