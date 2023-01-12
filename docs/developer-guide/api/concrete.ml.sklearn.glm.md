<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.glm`

Implement sklearn's Generalized Linear Models (GLM).

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PoissonRegressor`

A Poisson regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on PoissonRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: 'Union[int, dict]' = 8,
    alpha: 'float' = 1.0,
    fit_intercept: 'bool' = True,
    max_iter: 'int' = 100,
    tol: 'float' = 0.0001,
    warm_start: 'bool' = False,
    verbose: 'int' = 0
)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(
    y_preds: 'ndarray',
    already_dequantized: 'bool' = False
) → ndarray
```

Post-processing the predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The predictions to post-process.
- <b>`already_dequantized`</b> (bool):  Whether the inputs were already dequantized or not. Default  to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: 'ndarray', execute_in_fhe: 'bool' = False) → ndarray
```

Predict on user data.

Predict on user data using either the quantized clear model, implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute the inference in FHE. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The model's predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GammaRegressor`

A Gamma regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on GammaRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: 'Union[int, dict]' = 8,
    alpha: 'float' = 1.0,
    fit_intercept: 'bool' = True,
    max_iter: 'int' = 100,
    tol: 'float' = 0.0001,
    warm_start: 'bool' = False,
    verbose: 'int' = 0
)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(
    y_preds: 'ndarray',
    already_dequantized: 'bool' = False
) → ndarray
```

Post-processing the predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The predictions to post-process.
- <b>`already_dequantized`</b> (bool):  Whether the inputs were already dequantized or not. Default  to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: 'ndarray', execute_in_fhe: 'bool' = False) → ndarray
```

Predict on user data.

Predict on user data using either the quantized clear model, implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute the inference in FHE. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The model's predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TweedieRegressor`

A Tweedie regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on TweedieRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: 'Union[int, dict]' = 8,
    power: 'float' = 0.0,
    alpha: 'float' = 1.0,
    fit_intercept: 'bool' = True,
    link: 'str' = 'auto',
    max_iter: 'int' = 100,
    tol: 'float' = 0.0001,
    warm_start: 'bool' = False,
    verbose: 'int' = 0
)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(
    y_preds: 'ndarray',
    already_dequantized: 'bool' = False
) → ndarray
```

Post-processing the predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The predictions to post-process.
- <b>`already_dequantized`</b> (bool):  Whether the inputs were already dequantized or not. Default  to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/glm.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: 'ndarray', execute_in_fhe: 'bool' = False) → ndarray
```

Predict on user data.

Predict on user data using either the quantized clear model, implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute the inference in FHE. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The model's predictions.
