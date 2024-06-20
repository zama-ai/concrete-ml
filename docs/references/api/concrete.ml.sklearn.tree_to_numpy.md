<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.tree_to_numpy`

Implements the conversion of a tree model to a numpy function.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**
- **OPSET_VERSION_FOR_ONNX_EXPORT**
- **MSB_TO_KEEP_FOR_TREES**
- **MIN_CIRCUIT_THRESHOLD_FOR_TREES**

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_onnx_model`

```python
get_onnx_model(model, x: ndarray, framework: str) → ModelProto
```

Create ONNX model with Hummingbird convert method.

**Args:**

- <b>`model`</b>:  The tree model to convert.
- <b>`x`</b> (numpy.ndarray):  Dataset used to trace the tree inference and convert the model to ONNX.
- <b>`framework`</b> (str):  The framework from which the ONNX model is generated.
- <b>`(options`</b>:  'xgboost', 'sklearn')

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `workaround_squeeze_node_xgboost`

```python
workaround_squeeze_node_xgboost(onnx_model: ModelProto)
```

Workaround to fix torch issue that does not export the proper axis in the ONNX squeeze node.

FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2778 The squeeze ops does not have the proper dimensions. remove the following workaround when the issue is fixed Add the axis attribute to the Squeeze node

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assert_add_node_and_constant_in_xgboost_regressor_graph`

```python
assert_add_node_and_constant_in_xgboost_regressor_graph(onnx_model: ModelProto)
```

Assert if an Add node with a specific constant exists in the ONNX graph.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `add_transpose_after_last_node`

```python
add_transpose_after_last_node(
    onnx_model: ModelProto,
    fhe_ensembling: bool = False
)
```

Add transpose after last node.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The ONNX model.
- <b>`fhe_ensembling`</b> (bool):  Determines whether the sum of the trees' outputs is computed in FHE.  Default to False.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `preprocess_tree_predictions`

```python
preprocess_tree_predictions(
    init_tensor: ndarray,
    output_n_bits: int
) → QuantizedArray
```

Apply post-processing from the graph.

**Args:**

- <b>`init_tensor`</b> (numpy.ndarray):  Model parameters to be pre-processed.
- <b>`output_n_bits`</b> (int):  The number of bits of the output.

**Returns:**

- <b>`QuantizedArray`</b>:  Quantizer for the tree predictions.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tree_onnx_graph_preprocessing`

```python
tree_onnx_graph_preprocessing(
    onnx_model: ModelProto,
    framework: str,
    expected_number_of_outputs: int,
    fhe_ensembling: bool = False
)
```

Apply pre-processing onto the ONNX graph.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The ONNX model.
- <b>`framework`</b> (str):  The framework from which the ONNX model is generated.
- <b>`(options`</b>:  'xgboost', 'sklearn')
- <b>`expected_number_of_outputs`</b> (int):  The expected number of outputs in the ONNX model.
- <b>`fhe_ensembling`</b> (bool):  Determines whether the sum of the trees' outputs is computed in FHE.  Default to False.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tree_values_preprocessing`

```python
tree_values_preprocessing(
    onnx_model: ModelProto,
    framework: str,
    output_n_bits: int
) → QuantizedArray
```

Pre-process tree values.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The ONNX model.
- <b>`framework`</b> (str):  The framework from which the ONNX model is generated.
- <b>`(options`</b>:  'xgboost', 'sklearn')
- <b>`output_n_bits`</b> (int):  The number of bits of the output.

**Returns:**

- <b>`QuantizedArray`</b>:  Quantizer for the tree predictions.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L336"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tree_to_numpy`

```python
tree_to_numpy(
    model: BaseEstimator,
    x: ndarray,
    framework: str,
    use_rounding: bool = True,
    fhe_ensembling: bool = False,
    output_n_bits: int = 8
) → Tuple[Callable, List[UniformQuantizer], ModelProto]
```

Convert the tree inference to a numpy functions using Hummingbird.

**Args:**

- <b>`model`</b> (Callable):  The tree model to convert.
- <b>`x`</b> (numpy.ndarray):  The input data.
- <b>`use_rounding`</b> (bool):  Determines whether the rounding feature is enabled or disabled.  Default to True.
- <b>`fhe_ensembling`</b> (bool):  Determines whether the sum of the trees' outputs is computed in FHE.  Default to False.
- <b>`framework`</b> (str):  The framework from which the ONNX model is generated.
- <b>`(options`</b>:  'xgboost', 'sklearn')
- <b>`output_n_bits`</b> (int):  The number of bits of the output. Default to 8.

**Returns:**

- <b>`Tuple[Callable, List[QuantizedArray], onnx.ModelProto]`</b>:  A tuple with a function that takes a  numpy array and returns a numpy array, QuantizedArray object to quantize and de-quantize  the output of the tree, and the ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/tree_to_numpy.py#L530"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `onnx_fp32_model_to_quantized_model`

```python
onnx_fp32_model_to_quantized_model(
    onnx_model: ModelProto,
    n_bits: int,
    framework: str,
    expected_number_of_outputs: int,
    n_features: int,
    model_inputs: Optional[ndarray] = None
)
```

Build a FHE-compliant onnx-model using a fitted scikit-learn model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The fitted scikit-learn as a Hummingbird onnx model to convert
- <b>`n_bits`</b> (int):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters
- <b>`framework`</b> (str):  either sklearn or xgboost
- <b>`expected_number_of_outputs`</b> (int):  expected number of outputs
- <b>`n_features`</b> (int):  number of features as inputs of the model
- <b>`model_inputs`</b> (Optional\[numpy.ndarray\]):  optional dataset to use for quantization

**Returns:**

- <b>`onnx.ModelProto`</b>:  The converted onnx model
- <b>`Optional[Tuple[int, int]]`</b>:  Least significant bits to remove
- <b>`list[UniformQuantizer]`</b>:  inputs quantizers
- <b>`list[UniformQuantizer]`</b>:  outputs quantizers
