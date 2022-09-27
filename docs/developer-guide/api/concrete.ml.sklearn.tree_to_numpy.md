<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree_to_numpy.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.tree_to_numpy`

Implements the conversion of a tree model to a numpy function.

## **Global Variables**

- **MAXIMUM_TLU_BIT_WIDTH**
- **OPSET_VERSION_FOR_ONNX_EXPORT**
- **EXPECTED_NUMBER_OF_OUTPUTS_PER_TASK**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree_to_numpy.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tree_to_numpy`

```python
tree_to_numpy(
    model: ModelProto,
    x: ndarray,
    framework: str,
    task: Task,
    output_n_bits: Optional[int] = 8
) → Tuple[Callable, List[UniformQuantizer], ModelProto]
```

Convert the tree inference to a numpy functions using Hummingbird.

**Args:**

- <b>`model`</b> (onnx.ModelProto):  The model to convert.
- <b>`x`</b> (numpy.ndarray):  The input data.
- <b>`framework`</b> (str):  The framework from which the onnx_model is generated.
- <b>`(options`</b>:  'xgboost', 'sklearn')
- <b>`task`</b> (Task):  The task the model is solving
- <b>`output_n_bits`</b> (int):  The number of bits of the output.

**Returns:**

- <b>`Tuple[Callable, List[QuantizedArray], onnx.ModelProto]`</b>:  A tuple with a function that takes a  numpy array and returns a numpy array, QuantizedArray object to quantize and de-quantize  the output of the tree, and the ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/tree_to_numpy.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Task`

Task enumerate.
