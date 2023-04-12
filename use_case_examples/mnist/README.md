# MNIST With FHE

The `mnist_in_fhe.ipynb` notebook shows how to use Concrete ML to train a custom neural network
on the MNIST data-set.

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

- Then, create a directory for check_points

<!--pytest-codeblocks:skip-->

```bash
mkdir -p .checkpoints
```
