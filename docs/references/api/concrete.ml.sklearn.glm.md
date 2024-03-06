<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/glm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.glm`

Implement sklearn's Generalized Linear Models (GLM).

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PoissonRegressor`

A Poisson regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on PoissonRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html

<a href="../../../src/concrete/ml/sklearn/glm.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/sklearn/glm.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict')
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GammaRegressor`

A Gamma regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on GammaRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html

<a href="../../../src/concrete/ml/sklearn/glm.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/sklearn/glm.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict')
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TweedieRegressor`

A Tweedie regression model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

For more details on TweedieRegressor please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html

<a href="../../../src/concrete/ml/sklearn/glm.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/sklearn/glm.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict')
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/glm.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```
