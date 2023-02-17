<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/rf.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.rf`

Implement RandomForest models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/rf.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RandomForestClassifier`

Implements the RandomForest classifier.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/rf.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: int = 6,
    n_estimators=20,
    criterion='gini',
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None
)
```

Initialize the RandomForestClassifier.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/rf.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RandomForestRegressor`

Implements the RandomForest regressor.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/rf.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: int = 6,
    n_estimators=20,
    criterion='squared_error',
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None
)
```

Initialize the RandomForestRegressor.

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
