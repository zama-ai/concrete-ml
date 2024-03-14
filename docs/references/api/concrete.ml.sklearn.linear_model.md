<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.linear_model`

Implement sklearn linear model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearRegression`

A linear regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on LinearRegression please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    n_jobs=None,
    positive=False
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SGDClassifier`

An FHE linear classifier model fitted with stochastic gradient descent.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.
- <b>`fit_encrypted`</b> (bool):  Indicate if the model should be fitted in FHE or not. Default to  False.
- <b>`parameters_range`</b> (Optional\[Tuple\[float, float\]\]):  Range of values to consider for the  model's parameters when compiling it after training it in FHE (if fit_encrypted is set  to True). Default to None.
- <b>`batch_size`</b> (int):  Batch size to consider for the gradient descent during FHE training (if  fit_encrypted is set to True). Default to 8.

For more details on SGDClassifier please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    fit_encrypted=False,
    parameters_range=None,
    loss='log_loss',
    penalty='l2',
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter: int = 1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    n_jobs=None,
    random_state=None,
    learning_rate='optimal',
    eta0=0.0,
    power_t=0.5,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    warm_start=False,
    average=False
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> n_classes\_

Get the model's number of classes.

Using this attribute is deprecated.

**Returns:**

- <b>`int`</b>:  The model's number of classes.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

#### <kbd>property</kbd> target_classes\_

Get the model's classes.

Using this attribute is deprecated.

**Returns:**

- <b>`Optional[numpy.ndarray]`</b>:  The model's classes.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L953"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L592"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(
    X: Union[ndarray, Tensor, ForwardRef('DataFrame'), List],
    y: Union[ndarray, Tensor, ForwardRef('DataFrame'), ForwardRef('Series'), List],
    fhe: Optional[str, FheMode] = None,
    coef_init: Optional[ndarray] = None,
    intercept_init: Optional[ndarray] = None,
    sample_weight: Optional[ndarray] = None
)
```

Fit SGDClassifier.

For more details on some of these arguments please refer to: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html Training with encrypted data differs a bit from what is done by scikit-learn on multiple points:

- The learning rate used is constant (self.learning_rate_value)
- There is a batch size, it does not use the full dataset (self.batch_size)

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`fhe`</b> (Optional\[Union\[str, FheMode\]\]):  The mode to use for FHE training.  Can be FheMode.DISABLE for Concrete ML Python (quantized) training,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values. If None, training is  done in floating points in the clear through scikit-learn. Default to None.
- <b>`coef_init`</b> (Optional\[numpy.ndarray\]):  The initial coefficients to warm-start the  optimization. Default to None.
- <b>`intercept_init`</b> (Optional\[numpy.ndarray\]):  The initial intercept to warm-start the  optimization. Default to None.
- <b>`sample_weight`</b> (Optional\[numpy.ndarray\]):  Weights applied to individual samples (1. for  unweighted). It is currently not supported for FHE training. Default to None.

**Returns:**
The fitted estimator.

**Raises:**

- <b>`ValueError`</b>:  if `fhe` is provided but `fit_encrypted==False`
- <b>`NotImplementedError`</b>:  If parameter a 'sample_weight' is given while FHE training is  enabled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: bool = True) → dict
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1010"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L807"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `partial_fit`

```python
partial_fit(X: ndarray, y: ndarray, fhe: Optional[str, FheMode] = None)
```

Fit SGDClassifier for a single iteration.

This function does one iteration of SGD training. Looping n_times over this function is equivalent to calling 'fit' with max_iter=n_times.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`fhe`</b> (Optional\[Union\[str, FheMode\]\]):  The mode to use for FHE training.  Can be FheMode.DISABLE for Concrete ML Python (quantized) training,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values. If None, training is  done in floating points in the clear through scikit-learn. Default to None.

**Raises:**

- <b>`NotImplementedError`</b>:  If FHE training is disabled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L871"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X: Union[ndarray, Tensor, ForwardRef('DataFrame'), List],
    fhe: Union[FheMode, str] = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Probability estimates.

This method is only available for log loss and modified Huber loss. Multiclass probability estimates are derived from binary (one-vs.-rest) estimates by simple normalization, as recommended by Zadrozny and Elkan.

Binary probability estimates for loss="modified_huber" are given by (clip(decision_function(X), -1, 1) + 1) / 2. For other loss functions it is necessary to perform proper probability calibration by wrapping the classifier with `sklearn.calibration.CalibratedClassifierCV` instead.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List. It mush have a shape of (n_samples, n_features).
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted class probabilities, with shape (n_samples, n_classes).

**Raises:**

- <b>`NotImplementedError`</b>:  If the given loss is not supported.

References: Zadrozny and Elkan, "Transforming classifier scores into multiclass probability estimates", SIGKDD'02,

- <b>`https`</b>: //dl.acm.org/doi/pdf/10.1145/775047.775151

The justification for the formula in the loss="modified_huber" case is in the appendix B in:

- <b>`http`</b>: //jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1073"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SGDRegressor`

An FHE linear regression model fitted with stochastic gradient descent.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on SGDRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1093"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    loss='squared_error',
    penalty='l2',
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    random_state=None,
    learning_rate='invscaling',
    eta0=0.01,
    power_t=0.25,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    warm_start=False,
    average=False
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ElasticNet`

An ElasticNet regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on ElasticNet please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    alpha=1.0,
    l1_ratio=0.5,
    fit_intercept=True,
    normalize='deprecated',
    precompute=False,
    max_iter=1000,
    copy_X=True,
    tol=0.0001,
    warm_start=False,
    positive=False,
    random_state=None,
    selection='cyclic'
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Lasso`

A Lasso regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on Lasso please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    alpha: float = 1.0,
    fit_intercept=True,
    normalize='deprecated',
    precompute=False,
    copy_X=True,
    max_iter=1000,
    tol=0.0001,
    warm_start=False,
    positive=False,
    random_state=None,
    selection='cyclic'
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Ridge`

A Ridge regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on Ridge please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1480"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    alpha: float = 1.0,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    max_iter=None,
    tol=0.001,
    solver='auto',
    positive=False,
    random_state=None
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1537"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1568"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LogisticRegression`

A logistic regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on LogisticRegression please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1588"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=8,
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> n_classes\_

Get the model's number of classes.

Using this attribute is deprecated.

**Returns:**

- <b>`int`</b>:  The model's number of classes.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

#### <kbd>property</kbd> target_classes\_

Get the model's classes.

Using this attribute is deprecated.

**Returns:**

- <b>`Optional[numpy.ndarray]`</b>:  The model's classes.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/linear_model.py#L1665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```
