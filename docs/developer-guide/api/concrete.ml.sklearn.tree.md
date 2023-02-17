<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.tree`

Implement DecisionTree models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DecisionTreeClassifier`

Implements the sklearn DecisionTreeClassifier.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha: float = 0.0,
    n_bits: int = 6
)
```

Initialize the DecisionTreeClassifier.

# noqa: DAR101

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DecisionTreeRegressor`

Implements the sklearn DecisionTreeClassifier.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    criterion='squared_error',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0,
    n_bits: int = 6
)
```

Initialize the DecisionTreeRegressor.

# noqa: DAR101

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.
