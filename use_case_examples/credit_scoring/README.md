# Credit Scoring in FHE

In this directory, we present two notebooks:
- CreditScoringWithGraphics.ipynb, which derives from an existing notebook https://www.kaggle.com/code/ajay1735/my-credit-scoring-model: here we modified only slightly, to replace scikit-learn models with our Concrete ML models, and fix some minor typos or issues with our formatting rules
- CreditScoring.ipynb, FIXME, which is a shorter notebook, with just the models and a comparison between scikit-learn models and Concrete ML models, in term of speed and accuracy

Remark that hmeq.csv file comes from https://www.kaggle.com/code/ajay1735/my-credit-scoring-model/input .

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
