<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.linear_model`

Implement sklearn linear model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearRegression`

A linear regression model with FHE.

**Arguments:**

- <b>`n_bits`</b> (int):  default is 2.
- <b>`use_sum_workaround`</b> (bool):  indicate if the sum workaround should be used or not. This
- <b>`feature is experimental and should be used carefully. Important note`</b>:  it only works for a LinearRegression model with N features, N a power of 2, for now. More information available in the QuantizedReduceSum operator. Default to False.

For more details on LinearRegression please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=2,
    use_sum_workaround=False,
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

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Get the input quantization function.

**Returns:**

- <b>`Callable `</b>:  function that quantizes the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, *args, **kwargs) → Any
```

Fit the FHE linear model.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

**Returns:**
Any

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ElasticNet`

An ElasticNet regression model with FHE.

**Arguments:**

- <b>`n_bits`</b> (int):  default is 2.

For more details on ElasticNet please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=2,
    alpha=1.0,
    l1_ratio=0.5,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    positive=False
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Get the input quantization function.

**Returns:**

- <b>`Callable `</b>:  function that quantizes the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Lasso`

A Lasso regression model with FHE.

**Arguments:**

- <b>`n_bits`</b> (int):  default is 2.

For more details on Lasso please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=2,
    alpha: float = 1.0,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    positive=False
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Get the input quantization function.

**Returns:**

- <b>`Callable `</b>:  function that quantizes the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Ridge`

A Ridge regression model with FHE.

**Arguments:**

- <b>`n_bits`</b> (int):  default is 2.

For more details on Ridge please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=2,
    alpha: float = 1.0,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    positive=False
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Get the input quantization function.

**Returns:**

- <b>`Callable `</b>:  function that quantizes the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LogisticRegression`

A logistic regression model with FHE.

**Arguments:**

- <b>`n_bits`</b> (int):  default is 2.

For more details on LogisticRegression please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/sklearn/linear_model.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=2,
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

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Get the input quantization function.

**Returns:**

- <b>`Callable `</b>:  function that quantizes the input
