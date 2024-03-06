<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/xgb.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.xgb`

Implements XGBoost models.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/xgb.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `XGBClassifier`

Implements the XGBoost classifier.

See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn for more information about the parameters used.

<a href="../../../src/concrete/ml/sklearn/xgb.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: Union[int, Dict[str, int]] = 6,
    max_depth: Optional[int] = 3,
    learning_rate: Optional[float] = None,
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
    random_state: Optional[int] = None,
    verbosity: Optional[int] = None
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

<a href="../../../src/concrete/ml/sklearn/xgb.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/xgb.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/xgb.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `XGBRegressor`

Implements the XGBoost regressor.

See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn for more information about the parameters used.

<a href="../../../src/concrete/ml/sklearn/xgb.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    n_bits: Union[int, Dict[str, int]] = 6,
    max_depth: Optional[int] = 3,
    learning_rate: Optional[float] = None,
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
    random_state: Optional[int] = None,
    verbosity: Optional[int] = None
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

<a href="../../../src/concrete/ml/sklearn/xgb.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/xgb.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, *args, **kwargs) → Any
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/xgb.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/xgb.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```
