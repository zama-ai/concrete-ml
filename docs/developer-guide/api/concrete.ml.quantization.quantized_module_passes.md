<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_module_passes`

Optimization passes for QuantizedModules.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PowerOfTwoScalingRoundPBSAdapter`

Detect neural network patterns that can be optimized with round PBS.

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(qmodule: QuantizedModule) → None
```

______________________________________________________________________

#### <kbd>property</kbd> num_ignored_valid_patterns

Get the number of optimizable patterns that were ignored.

Patterns could be ignored since a number of rounding bits was set manually through the compilation function.

**Returns:**

- <b>`result`</b> (int):  number of patterns that could be optimized but were not

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_op_predecessors`

```python
compute_op_predecessors() → DefaultDict[Union[QuantizedOp, NoneType], List[Tuple[Union[QuantizedOp, NoneType], str]]]
```

Compute the predecessors for each QuantizedOp in a QuantizedModule.

Stores, for each quantized op, a list of quantized ops that produce its inputs. Currently only the first input of the operations is considered as it is, usually, the encrypted input.

**Returns:**

- <b>`result`</b> (PredecessorsType):  a dictionary containing a hierarchy of op  predecessors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `detect_patterns`

```python
detect_patterns(
    predecessors: DefaultDict[Optional[QuantizedOp], List[Tuple[Optional[QuantizedOp], str]]]
) → Dict[QuantizedMixingOp, Tuple[List[Union[QuantizedOp, NoneType]], Union[QuantizedOp, NoneType]]]
```

Detect the patterns that can be optimized with roundPBS in the QuantizedModule.

**Args:**

- <b>`predecessors`</b> (PredecessorsType):  Module predecessor operation list

**Returns:**

- <b>`result`</b> (PatternDict):  list of optimizable patterns

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `match_path_pattern`

```python
match_path_pattern(
    predecessors: DefaultDict[Optional[QuantizedOp], List[Tuple[Optional[QuantizedOp], str]]],
    nodes_in_path: List[Optional[QuantizedOp]],
    input_producer_of_path: Optional[QuantizedOp]
) → bool
```

Determine if a pattern has the structure that makes it viable for roundPBS.

**Args:**

- <b>`predecessors`</b> (PredecessorsType):  Module predecessor operation list
- <b>`nodes_in_path`</b> (List\[QuantizedOp\]):  list of quantized ops in the pattern
- <b>`input_producer_of_path`</b> (Optional\[QuantizedOp\]):  operation that produces the input

**Returns:**

- <b>`result`</b> (bool):  whether the pattern can be optimized

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process`

```python
process() → Dict[QuantizedMixingOp, Tuple[List[Union[QuantizedOp, NoneType]], Union[QuantizedOp, NoneType]]]
```

Analyze an ONNX graph and detect Gemm/Conv patterns that can use RoundPBS.

We want to detect a gemm/conv node whose weights/bias are Brevitas QAT, and whose input is produced by a Brevitas QAT node that is applied on the output of another Gemm/conv node. Optionally a Relu can be placed before this input quantization node.

Nothing will be done if rounding is already specified.

**Returns:**

- <b>`result`</b> (PatternDict):  a dictionary containing for each Conv/Gemm node for which  round PBS can be applied based on power-of-two scaling factors

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module_passes.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process_patterns`

```python
process_patterns(
    valid_paths: Dict[QuantizedMixingOp, Tuple[List[Optional[QuantizedOp]], Optional[QuantizedOp]]]
) → Dict[QuantizedMixingOp, Tuple[List[Union[QuantizedOp, NoneType]], Union[QuantizedOp, NoneType]]]
```

Configure the rounding bits of roundPBS for the optimizable operations.

**Args:**

- <b>`valid_paths`</b> (PatternDict):  list of optimizable patterns

**Returns:**

- <b>`result`</b> (PatternDict):  list of patterns actually optimized with roundPBS
