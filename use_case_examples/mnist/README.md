# MNIST With FHE

## Running the application on your machine

In this directory, ie `mnist`, you can do the following steps.

### Do once

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

- Then, create a directory for check_points

<!--pytest-codeblocks:skip-->

```bash
mkdir -p .checkpoints
```

### Do each time you relaunch the application

<!--pytest-codeblocks:skip-->

```bash
source .venv/bin/activate
python use_case_examples/mnist/mnist_in_fhe.py
```

### If you want to use PyTorch with CUDA in Concrete-ML

```bash
pip3 install --upgrade torch==1.12.1 torchvision torchsummary --extra-index-url https://download.pytorch.org/whl/cu113
```

Remark that you may want to change some options in `mnist_in_fhe.py`, and notably:

- epochs: the number of epoch, default is 1
- sparsity: the sparsity of the model, default is 4
- quantization_bits: the number of bits for quantization, default is 2
- do_test_in_fhe: whether we perform tests in FHE, default is True
- do_training: whether we do the training, default is True
