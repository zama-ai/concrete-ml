<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/svm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.svm`

Implement Support Vector Machine.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/svm.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearSVR`

A Regression Support Vector Machine (SVM).

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on LinearSVR please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/svm.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    epsilon=0.0,
    tol=0.0001,
    C=1.0,
    loss='epsilon_insensitive',
    fit_intercept=True,
    intercept_scaling=1.0,
    dual=True,
    verbose=0,
    random_state=None,
    max_iter=1000
)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/svm.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearSVC`

A Classification Support Vector Machine (SVM).

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on LinearSVC please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/svm.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    penalty='l2',
    loss='squared_hinge',
    dual=True,
    tol=0.0001,
    C=1.0,
    multi_class='ovr',
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    verbose=0,
    random_state=None,
    max_iter=1000
)
```
