<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/onnx/onnx_impl_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.onnx_impl_utils`

Utility functions for onnx operator implementations.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/onnx/onnx_impl_utils.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_onnx_pad`

```python
numpy_onnx_pad(
    x: ndarray,
    pads: Tuple[int, ],
    pad_value: Union[float, int, ndarray] = 0,
    int_only: bool = False
) → ndarray
```

Pad a tensor according to ONNX spec, using an optional custom pad value.

**Args:**

- <b>`x`</b> (numpy.ndarray):  input tensor to pad
- <b>`pads`</b> (List\[int\]):  padding values according to ONNX spec
- <b>`pad_value`</b> (Optional\[Union\[float, int\]\]):  value used to fill in padding, default 0
- <b>`int_only`</b> (bool):  set to True to generate integer only code with Concrete-Numpy

**Returns:**

- <b>`res`</b> (numpy.ndarray):  the input tensor with padding applied

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/onnx/onnx_impl_utils.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/onnx/onnx_impl_utils.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

The ONNX standard uses ceil_mode=1 to match tensorflow style pooling output computation. In this setting, the kernel can be placed at a valid position even though it contains values outside of the input shape including padding. The ceil_mode parameter controls whether this mode is enabled. If the mode is not enabled, the output shape follows PyTorch rules.

**Args:**

- <b>`input_shape`</b> (Tuple\[int, ...\]):  shape of the input to be padded as N x C x H x W
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  shape of the conv or pool kernel, as Kh x Kw (or n-d)
- <b>`pads`</b> (Tuple\[int, ...\]):  padding values following ONNX spec:  dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end  where in the 2-d case dim1 is H, dim2 is W
- <b>`strides`</b> (Tuple\[int, ...\]):  strides for each dimension
- <b>`ceil_mode`</b> (int):  set to 1 to use the `ceil` function to compute the output shape, as  described in the PyTorch doc

**Returns:**

- <b>`res`</b> (Tuple\[int, ...\]):  shape of the output of a conv or pool operator with given parameters

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/onnx/onnx_impl_utils.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `onnx_avgpool_compute_norm_const`

```python
onnx_avgpool_compute_norm_const(
    input_shape: Tuple[int, ],
    kernel_shape: Tuple[int, ],
    pads: Tuple[int, ],
    strides: Tuple[int, ],
    ceil_mode: int
) → Union[ndarray, float]
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
