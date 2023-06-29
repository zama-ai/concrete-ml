# Healthcare diagnosis

Give a diagnosis using FHE to preserve the privacy of the patient. We show how
to train several models using Concrete ML and choose the one that provides
the best accuracy with the lowest inference time. Finally, we export the model
in order to use it in the [Hugging Face space](https://huggingface.co/spaces/zama-fhe/health_prediction/)
which provides a live interactive demo this model in use.

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
