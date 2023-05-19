# Credit Scoring in FHE

In this directory, CreditScoring.ipynb compares scikit-learn models and Concrete ML models, in term of speed and accuracy, on a credit scoring task. This notebook was inspired by an existing notebook https://www.kaggle.com/code/ajay1735/my-credit-scoring-model. The data-set hmeq.csv file comes from https://www.kaggle.com/code/ajay1735/my-credit-scoring-model/input .

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
