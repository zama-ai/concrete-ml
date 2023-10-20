<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.neighbors`

Implement sklearn neighbors model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `KNeighborsClassifier`

A k-nearest neighbors classifier model with FHE.

**Parameters:**

- <b>`n_bits`</b> (int):  Number of bits to quantize the model. The value will be used for quantizing  inputs and X_fit. Default to 3.

For more details on KNeighborsClassifier please refer to the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits=2,
    n_neighbors=3,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None
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

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `kneighbors`

```python
kneighbors(X: Union[ndarray, Tensor, ForwardRef('DataFrame'), List]) → ndarray
```

Return the knearest distances and their respective indices for each query point.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.

**Raises:**

- <b>`NotImplementedError`</b>:  The method is not implemented for now.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/neighbors.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X: Union[ndarray, Tensor, ForwardRef('DataFrame'), List],
    fhe: Union[FheMode, str] = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict class probabilities.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Raises:**

- <b>`NotImplementedError`</b>:  The method is not implemented for now.
