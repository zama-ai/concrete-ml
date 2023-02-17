<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.xgb`

Implements XGBoost models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `XGBClassifier`

Implements the XGBoost classifier.

See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn for more information about the parameters used.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: int = 6,
    max_depth: Optional[int] = 3,
    learning_rate: Optional[float] = 0.1,
    n_estimators: Optional[int] = 20,
    objective: Optional[str] = 'binary:logistic',
    booster: Optional[str] = None,
    tree_method: Optional[str] = None,
    n_jobs: Optional[int] = None,
    gamma: Optional[float] = None,
    min_child_weight: Optional[float] = None,
    max_delta_step: Optional[float] = None,
    subsample: Optional[float] = None,
    colsample_bytree: Optional[float] = None,
    colsample_bylevel: Optional[float] = None,
    colsample_bynode: Optional[float] = None,
    reg_alpha: Optional[float] = None,
    reg_lambda: Optional[float] = None,
    scale_pos_weight: Optional[float] = None,
    base_score: Optional[float] = None,
    missing: float = nan,
    num_parallel_tree: Optional[int] = None,
    monotone_constraints: Optional[Dict[str, int], str] = None,
    interaction_constraints: Optional[str, List[Tuple[str]]] = None,
    importance_type: Optional[str] = None,
    gpu_id: Optional[int] = None,
    validate_parameters: Optional[bool] = None,
    predictor: Optional[str] = None,
    enable_categorical: bool = False,
    use_label_encoder: bool = False,
    random_state: Optional[RandomState, int] = None,
    verbosity: Optional[int] = None
)
```

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `XGBRegressor`

Implements the XGBoost regressor.

See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn for more information about the parameters used.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: int = 6,
    max_depth: Optional[int] = 3,
    learning_rate: Optional[float] = 0.1,
    n_estimators: Optional[int] = 20,
    objective: Optional[str] = 'reg:squarederror',
    booster: Optional[str] = None,
    tree_method: Optional[str] = None,
    n_jobs: Optional[int] = None,
    gamma: Optional[float] = None,
    min_child_weight: Optional[float] = None,
    max_delta_step: Optional[float] = None,
    subsample: Optional[float] = None,
    colsample_bytree: Optional[float] = None,
    colsample_bylevel: Optional[float] = None,
    colsample_bynode: Optional[float] = None,
    reg_alpha: Optional[float] = None,
    reg_lambda: Optional[float] = None,
    scale_pos_weight: Optional[float] = None,
    base_score: Optional[float] = None,
    missing: float = nan,
    num_parallel_tree: Optional[int] = None,
    monotone_constraints: Optional[Dict[str, int], str] = None,
    interaction_constraints: Optional[str, List[Tuple[str]]] = None,
    importance_type: Optional[str] = None,
    gpu_id: Optional[int] = None,
    validate_parameters: Optional[bool] = None,
    predictor: Optional[str] = None,
    enable_categorical: bool = False,
    use_label_encoder: bool = False,
    random_state: Optional[RandomState, int] = None,
    verbosity: Optional[int] = None
)
```

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **kwargs) → Any
```

Fit the tree-based estimator.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`**kwargs`</b>:  args for super().fit

**Returns:**

- <b>`Any`</b>:  The fitted model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/xgb.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```
