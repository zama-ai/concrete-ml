<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/onnx/onnx_impl_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.onnx_impl_utils`

Utility functions for onnx operator implementations.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_impl_utils.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_onnx_pad`

```python
numpy_onnx_pad(
    x: ndarray,
    pads: Tuple[int, ],
    pad_value: Union[float, int, ndarray] = 0,
    int_only: bool = False
) → Union[ndarray, Tracer]
```

Pad a tensor according to ONNX spec, using an optional custom pad value.

**Args:**

- <b>`x`</b> (numpy.ndarray):  input tensor to pad
- <b>`pads`</b> (List\[int\]):  padding values according to ONNX spec
- <b>`pad_value`</b> (Optional\[Union\[float, int\]\]):  value used to fill in padding, default 0
- <b>`int_only`</b> (bool):  set to True to generate integer only code with Concrete

**Returns:**

- <b>`res`</b> (numpy.ndarray):  the input tensor with padding applied

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_impl_utils.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_conv_output_dims`

```python
compute_conv_output_dims(
    input_shape: Tuple[int, ],
    kernel_shape: Tuple[int, ],
    pads: Tuple[int, ],
    strides: Tuple[int, ],
    ceil_mode: int
) → Tuple[int, ]
```

Compute the output shape of a pool or conv operation.

See https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html for details on the computation of the output shape.

**Args:**

- <b>`input_shape`</b> (Tuple\[int, ...\]):  shape of the input to be padded as N x C x H x W
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  shape of the conv or pool kernel, as Kh x Kw (or n-d)
- <b>`pads`</b> (Tuple\[int, ...\]):  padding values following ONNX spec:  dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end  where in the 2-d case dim1 is H, dim2 is W
- <b>`strides`</b> (Tuple\[int, ...\]):  strides for each dimension
- <b>`ceil_mode`</b> (int):  set to 1 to use the `ceil` function to compute the output shape, as  described in the PyTorch doc

**Returns:**

- <b>`res`</b> (Tuple\[int, ...\]):  shape of the output of a conv or pool operator with given parameters

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_impl_utils.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_onnx_pool_padding`

```python
compute_onnx_pool_padding(
    input_shape: Tuple[int, ],
    kernel_shape: Tuple[int, ],
    pads: Tuple[int, ],
    strides: Tuple[int, ],
    ceil_mode: int
) → Tuple[int, ]
```

Compute any additional padding needed to compute pooling layers.

The ONNX standard uses ceil_mode=1 to match TensorFlow style pooling output computation. In this setting, the kernel can be placed at a valid position even though it contains values outside of the input shape including padding. The ceil_mode parameter controls whether this mode is enabled. If the mode is not enabled, the output shape follows PyTorch rules.

**Args:**

- <b>`input_shape`</b> (Tuple\[int, ...\]):  shape of the input to be padded as N x C x H x W
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  shape of the conv or pool kernel, as Kh x Kw (or n-d)
- <b>`pads`</b> (Tuple\[int, ...\]):  padding values following ONNX spec:  dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end  where in the 2-d case dim1 is H, dim2 is W
- <b>`strides`</b> (Tuple\[int, ...\]):  strides for each dimension
- <b>`ceil_mode`</b> (int):  set to 1 to use the `ceil` function to compute the output shape, as  described in the PyTorch doc

**Returns:**

- <b>`res`</b> (Tuple\[int, ...\]):  shape of the output of a conv or pool operator with given parameters

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_impl_utils.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `onnx_avgpool_compute_norm_const`

```python
onnx_avgpool_compute_norm_const(
    input_shape: Tuple[int, ],
    kernel_shape: Tuple[int, ],
    pads: Tuple[int, ],
    strides: Tuple[int, ],
    ceil_mode: int
) → Union[ndarray, float, Tracer]
```

Compute the average pooling normalization constant.

This constant can be a tensor of the same shape as the input or a scalar.

**Args:**

- <b>`input_shape`</b> (Tuple\[int, ...\]):  shape of the input to be padded as N x C x H x W
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  shape of the conv or pool kernel, as Kh x Kw (or n-d)
- <b>`pads`</b> (Tuple\[int, ...\]):  padding values following ONNX spec:  dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end  where in the 2-d case dim1 is H, dim2 is W
- <b>`strides`</b> (Tuple\[int, ...\]):  strides for each dimension
- <b>`ceil_mode`</b> (int):  set to 1 to use the `ceil` function to compute the output shape, as  described in the PyTorch doc

**Returns:**

- <b>`res`</b> (float):  tensor or scalar, corresponding to normalization factors to apply for the  average pool computation for each valid kernel position

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_impl_utils.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rounded_comparison`

```python
rounded_comparison(
    x: ndarray,
    y: ndarray,
    lsbs_to_remove: int,
    operation: Callable[[int], bool]
) → Tuple[bool]
```

Comparison operation using `round_bit_pattern` function.

`round_bit_pattern` rounds the bit pattern of an integer to the closer It also checks for any potential overflow. If so, it readjusts the LSBs accordingly.

The parameter `lsbs_to_remove` in `round_bit_pattern` can either be an integer specifying the number of LSBS to remove, or an `AutoRounder` object that determines the required number of LSBs based on the specified number of MSBs to retain. But in our case, we choose to compute the LSBs manually.

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor
- <b>`lsbs_to_remove`</b> (int):  Number of the least significant bits to remove
- <b>`operation`</b> (ComparisonOperationType):  Comparison operation, which can `<`, `<=` and `==`

**Returns:**

- <b>`Tuple[bool]`</b>:  If x and y satisfy the comparison operator.
