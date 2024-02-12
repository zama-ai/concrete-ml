<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.ops_impl`

ONNX ops implementation in Python + NumPy.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `cast_to_float`

```python
cast_to_float(inputs)
```

Cast values to floating points.

**Args:**

- <b>`inputs`</b> (Tuple\[numpy.ndarray\]):  The values to consider.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  The float values.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `onnx_func_raw_args`

```python
onnx_func_raw_args(*args, output_is_raw: bool = False)
```

Decorate a numpy onnx function to flag the raw/non quantized inputs.

**Args:**

- <b>`*args (tuple[Any])`</b>:  function argument names
- <b>`output_is_raw`</b> (bool):  marks the function as returning raw  values that should not be quantized

**Returns:**

- <b>`result`</b> (ONNXMixedFunction):  wrapped numpy function with a list of mixed arguments

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_where_body`

```python
numpy_where_body(c: ndarray, t: ndarray, f: Union[ndarray, int]) → ndarray
```

Compute the equivalent of numpy.where.

This function is not mapped to any ONNX operator (as opposed to numpy_where). It is usable by functions which are mapped to ONNX operators, e.g., numpy_div or numpy_where.

**Args:**

- <b>`c`</b> (numpy.ndarray):  Condition operand.
- <b>`t`</b> (numpy.ndarray):  True operand.
- <b>`f`</b> (numpy.ndarray):  False operand.

**Returns:**

- <b>`numpy.ndarray`</b>:  numpy.where(c, t, f)

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_where`

```python
numpy_where(c: ndarray, t: ndarray, f: ndarray) → Tuple[ndarray]
```

Compute the equivalent of numpy.where.

**Args:**

- <b>`c`</b> (numpy.ndarray):  Condition operand.
- <b>`t`</b> (numpy.ndarray):  True operand.
- <b>`f`</b> (numpy.ndarray):  False operand.

**Returns:**

- <b>`numpy.ndarray`</b>:  numpy.where(c, t, f)

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_add`

```python
numpy_add(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute add in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-13

**Args:**

- <b>`a`</b> (numpy.ndarray):  First operand.
- <b>`b`</b> (numpy.ndarray):  Second operand.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Result, has same element type as two inputs

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_constant`

```python
numpy_constant(**kwargs)
```

Return the constant passed as a kwarg.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-13

**Args:**

- <b>`**kwargs`</b>:  keyword arguments

**Returns:**

- <b>`Any`</b>:  The stored constant.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_gemm`

```python
numpy_gemm(
    a: ndarray,
    b: ndarray,
    c: Optional[ndarray] = None,
    alpha: float = 1,
    beta: float = 1,
    transA: int = 0,
    transB: int = 0
) → Tuple[ndarray]
```

Compute Gemm in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-13

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M)  if transA is non-zero.
- <b>`b`</b> (numpy.ndarray):  Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K)  if transB is non-zero.
- <b>`c`</b> (Optional\[numpy.ndarray\]):  Optional input tensor C. If not specified, the  computation is done as if C is a scalar 0. The shape of C should be unidirectional  broadcastable to (M, N).  Defaults to None.
- <b>`alpha`</b> (float):  Scalar multiplier for the product of input tensors A * B.  Defaults to 1.
- <b>`beta`</b> (float):  Scalar multiplier for input tensor C.  Defaults to 1.
- <b>`transA`</b> (int):  Whether A should be transposed. The type is kept as int as it is the  type used by ONNX and it can easily be interpreted by Python as a boolean.  Defaults to 0.
- <b>`transB`</b> (int):  Whether B should be transposed. The type is kept as int as it is the  type used by ONNX and it can easily be interpreted by Python as a boolean.  Defaults to 0.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  The tuple containing the result tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_matmul`

```python
numpy_matmul(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute matmul in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-13

**Args:**

- <b>`a`</b> (numpy.ndarray):  N-dimensional matrix A
- <b>`b`</b> (numpy.ndarray):  N-dimensional matrix B

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Matrix multiply results from A * B

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_relu`

```python
numpy_relu(x: ndarray) → Tuple[ndarray]
```

Compute relu in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-14

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_sigmoid`

```python
numpy_sigmoid(x: ndarray) → Tuple[ndarray]
```

Compute sigmoid in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sigmoid-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L346"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_softmax`

```python
numpy_softmax(x, axis=1, keepdims=True)
```

Compute softmax in numpy according to ONNX spec.

Softmax is currently not supported in FHE.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#softmax-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`axis`</b> (None, int, tuple of int):  Axis or axes along which a softmax's sum is performed. If  None, it will sum all of the elements of the input array.  If axis is negative it counts  from the last to the first axis. Default to 1.
- <b>`keepdims`</b> (bool):  If True, the axes which are reduced along the sum are left in the result as  dimensions with size one. Default to True.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L369"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_cos`

```python
numpy_cos(x: ndarray) → Tuple[ndarray]
```

Compute cos in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cos-7

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L385"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_cosh`

```python
numpy_cosh(x: ndarray) → Tuple[ndarray]
```

Compute cosh in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cosh-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_sin`

```python
numpy_sin(x: ndarray) → Tuple[ndarray]
```

Compute sin in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sin-7

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_sinh`

```python
numpy_sinh(x: ndarray) → Tuple[ndarray]
```

Compute sinh in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sinh-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L433"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_tan`

```python
numpy_tan(x: ndarray) → Tuple[ndarray]
```

Compute tan in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tan-7

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_tanh`

```python
numpy_tanh(x: ndarray) → Tuple[ndarray]
```

Compute tanh in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tanh-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_acos`

```python
numpy_acos(x: ndarray) → Tuple[ndarray]
```

Compute acos in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acos-7

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L481"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_acosh`

```python
numpy_acosh(x: ndarray) → Tuple[ndarray]
```

Compute acosh in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acosh-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L497"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_asin`

```python
numpy_asin(x: ndarray) → Tuple[ndarray]
```

Compute asin in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asin-7

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_asinh`

```python
numpy_asinh(x: ndarray) → Tuple[ndarray]
```

Compute sinh in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asinh-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_atan`

```python
numpy_atan(x: ndarray) → Tuple[ndarray]
```

Compute atan in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atan-7

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_atanh`

```python
numpy_atanh(x: ndarray) → Tuple[ndarray]
```

Compute atanh in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atanh-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L561"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_elu`

```python
numpy_elu(x: ndarray, alpha: float = 1) → Tuple[ndarray]
```

Compute elu in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Elu-6

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`alpha`</b> (float):  Coefficient

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L577"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_selu`

```python
numpy_selu(
    x: ndarray,
    alpha: float = 1.6732632423543772,
    gamma: float = 1.0507009873554805
) → Tuple[ndarray]
```

Compute selu in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Selu-6

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`alpha`</b> (float):  Coefficient
- <b>`gamma`</b> (float):  Coefficient

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L599"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_celu`

```python
numpy_celu(x: ndarray, alpha: float = 1) → Tuple[ndarray]
```

Compute celu in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Celu-12

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`alpha`</b> (float):  Coefficient

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L615"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_leakyrelu`

```python
numpy_leakyrelu(x: ndarray, alpha: float = 0.01) → Tuple[ndarray]
```

Compute leakyrelu in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LeakyRelu-6

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`alpha`</b> (float):  Coefficient

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L631"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_thresholdedrelu`

```python
numpy_thresholdedrelu(x: ndarray, alpha: float = 1) → Tuple[ndarray]
```

Compute thresholdedrelu in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ThresholdedRelu-10

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`alpha`</b> (float):  Coefficient

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L650"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_hardsigmoid`

```python
numpy_hardsigmoid(
    x: ndarray,
    alpha: float = 0.2,
    beta: float = 0.5
) → Tuple[ndarray]
```

Compute hardsigmoid in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#HardSigmoid-6

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`alpha`</b> (float):  Coefficient
- <b>`beta`</b> (float):  Coefficient

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L669"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_softplus`

```python
numpy_softplus(x: ndarray) → Tuple[ndarray]
```

Compute softplus in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softplus-1

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L686"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_abs`

```python
numpy_abs(x: ndarray) → Tuple[ndarray]
```

Compute abs in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Abs-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L703"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_div`

```python
numpy_div(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute div in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-14

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L735"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_mul`

```python
numpy_mul(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute mul in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-14

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_sub`

```python
numpy_sub(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute sub in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-14

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L773"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_log`

```python
numpy_log(x: ndarray) → Tuple[ndarray]
```

Compute log in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L816"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_erf`

```python
numpy_erf(x: ndarray) → Tuple[ndarray]
```

Compute erf in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Erf-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L833"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_hardswish`

```python
numpy_hardswish(x: ndarray) → Tuple[ndarray]
```

Compute hardswish in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#hardswish-14

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L854"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_exp`

```python
numpy_exp(x: ndarray) → Tuple[ndarray]
```

Compute exponential in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Exp-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  The exponential of the input tensor computed element-wise

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L871"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_equal`

```python
numpy_equal(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute equal in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-11

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L891"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rounded_numpy_equal_for_trees`

```python
rounded_numpy_equal_for_trees(
    x: ndarray,
    y: ndarray,
    lsbs_to_remove_for_trees: Optional[int] = None
) → Tuple[ndarray]
```

Compute rounded equal in numpy according to ONNX spec for tree-based models only.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-11

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor
- <b>`lsbs_to_remove_for_trees`</b> (Optional\[int\]):  Number of the least significant bits to remove  for tree-based models only.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L928"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_equal_float`

```python
numpy_equal_float(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute equal in numpy according to ONNX spec and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L947"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_not`

```python
numpy_not(x: ndarray) → Tuple[ndarray]
```

Compute not in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L964"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_not_float`

```python
numpy_not_float(x: ndarray) → Tuple[ndarray]
```

Compute not in numpy according to ONNX spec and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L981"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_greater`

```python
numpy_greater(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute greater in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1000"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_greater_float`

```python
numpy_greater_float(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute greater in numpy according to ONNX spec and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1019"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_greater_or_equal`

```python
numpy_greater_or_equal(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute greater or equal in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1038"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_greater_or_equal_float`

```python
numpy_greater_or_equal_float(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute greater or equal in numpy according to ONNX specs and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1057"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_less`

```python
numpy_less(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute less in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1075"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rounded_numpy_less_for_trees`

```python
rounded_numpy_less_for_trees(
    x: ndarray,
    y: ndarray,
    lsbs_to_remove_for_trees: Optional[int] = None
) → Tuple[ndarray]
```

Compute rounded less in numpy according to ONNX spec for tree-based models only.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor
- <b>`lsbs_to_remove_for_trees`</b> (Optional\[int\]):  Number of the least significant bits to remove  for tree-based models only.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_less_float`

```python
numpy_less_float(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute less in numpy according to ONNX spec and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_less_or_equal`

```python
numpy_less_or_equal(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute less or equal in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rounded_numpy_less_or_equal_for_trees`

```python
rounded_numpy_less_or_equal_for_trees(
    x: ndarray,
    y: ndarray,
    lsbs_to_remove_for_trees: Optional[int] = None
) → Tuple[ndarray]
```

Compute rounded less or equal in numpy according to ONNX spec for tree-based models only.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor
- <b>`lsbs_to_remove_for_trees`</b> (Optional\[int\]):  Number of the least significant bits to remove  for tree-based models only.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_less_or_equal_float`

```python
numpy_less_or_equal_float(x: ndarray, y: ndarray) → Tuple[ndarray]
```

Compute less or equal in numpy according to ONNX spec and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`y`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_identity`

```python
numpy_identity(x: ndarray) → Tuple[ndarray]
```

Compute identity in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-14

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_transpose`

```python
numpy_transpose(x: ndarray, perm=None) → Tuple[ndarray]
```

Transpose in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Transpose-13

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor
- <b>`perm`</b> (numpy.ndarray):  Permutation of the axes

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_conv`

```python
numpy_conv(
    x: ndarray,
    w: ndarray,
    b: Optional[ndarray] = None,
    dilations: Tuple[int, ],
    group: int = 1,
    kernel_shape: Tuple[int, ],
    pads: Tuple[int, ],
    strides: Tuple[int, ]
) → Tuple[ndarray]
```

Compute N-D convolution using Torch.

Currently supports 2d convolution with torch semantics. This function is also ONNX compatible.

See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv

**Args:**

- <b>`x`</b> (numpy.ndarray):  input data (many dtypes are supported). Shape is N x C x H x W for 2d
- <b>`w`</b> (numpy.ndarray):  weights tensor. Shape is (O x I x Kh x Kw) for 2d
- <b>`b`</b> (Optional\[numpy.ndarray\]):  bias tensor, Shape is (O,). Default to None.
- <b>`dilations`</b> (Tuple\[int, ...\]):  dilation of the kernel, default 1 on all dimensions.
- <b>`group`</b> (int):  number of convolution groups, can be 1 or a multiple of both (C,) and (O,), so  that I = C / group. Default to 1.
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  shape of the kernel. Should have 2 elements for 2d conv
- <b>`pads`</b> (Tuple\[int, ...\]):  padding in ONNX format (begin, end) on each axis
- <b>`strides`</b> (Tuple\[int, ...\]):  stride of the convolution on each axis

**Returns:**

- <b>`res`</b> (numpy.ndarray):  a tensor of size (N x OutChannels x OutHeight x OutWidth).
- <b>`See https`</b>: //pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1337"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_avgpool`

```python
numpy_avgpool(
    x: ndarray,
    kernel_shape: Tuple[int, ],
    auto_pad: str = 'NOTSET',
    ceil_mode: int = 0,
    count_include_pad: int = 1,
    pads: Optional[Tuple[int, ]] = None,
    strides: Optional[Tuple[int, ]] = None
) → Tuple[ndarray]
```

Compute Average Pooling using Torch.

Currently supports 2d average pooling with torch semantics. This function is ONNX compatible.

See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input data of shape (N, C, H, W), as only 2D inputs are currently  supported.
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  The size of the kernel along each axis. Currently, only 2D  kernels are supported.
- <b>`auto_pad`</b> (str):  Only the default "NOTSET" value is currently supported, which means  explicit padding is used.
- <b>`ceil_mode`</b> (int):  Whether to use ONNX's ceil (1) or floor (0, the default) to compute the  output shape.
- <b>`count_include_pad`</b> (int):  Whether include pad pixels when calculating values for the edges.  Currently, setting this parameter to 0 is not supported in Concrete ML.
- <b>`pads`</b> (Tuple\[int, ...\]):  Padding for the beginning and ending along each spatial axis.  Expected format is \[x1_begin, x2_begin...x1_end, x2_end, ...\] where xi_begin (resp.  xi_end) is the number of pixels added at the beginning (resp. end) of axis `i`.
- <b>`strides`</b> (Tuple\[int, ...\]):  Stride along each spatial axis. If not present, the stride  defaults to 1 along each spatial axis.

**Returns:**

- <b>`res`</b> (numpy.ndarray):  a tensor of size (N x InChannels x OutHeight x OutWidth).
- <b>`See https`</b>: //pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_maxpool`

```python
numpy_maxpool(
    x: ndarray,
    kernel_shape: Tuple[int, ],
    strides: Tuple[int, ] = None,
    auto_pad: str = 'NOTSET',
    pads: Tuple[int, ] = None,
    dilations: Optional[Tuple[int, ], List[int]] = None,
    ceil_mode: int = 0,
    storage_order: int = 0
) → Tuple[ndarray]
```

Compute Max Pooling using Torch.

Currently supports 2d max pooling with torch semantics. This function is ONNX compatible.

See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool

**Args:**

- <b>`x`</b> (numpy.ndarray):  the input
- <b>`kernel_shape`</b> (Union\[Tuple\[int, ...\], List\[int\]\]):  shape of the kernel
- <b>`strides`</b> (Optional\[Union\[Tuple\[int, ...\], List\[int\]\]\]):  stride along each spatial axis  set to 1 along each spatial axis if not set
- <b>`auto_pad`</b> (str):  padding strategy, default = "NOTSET"
- <b>`pads`</b> (Optional\[Union\[Tuple\[int, ...\], List\[int\]\]\]):  padding for the beginning and ending  along each spatial axis (D1_begin, D2_begin, ..., D1_end, D2_end, ...)  set to 0 along each spatial axis if not set
- <b>`dilations`</b> (Optional\[Union\[Tuple\[int, ...\], List\[int\]\]\]):  dilation along each spatial axis  set to 1 along each spatial axis if not set
- <b>`ceil_mode`</b> (int):  ceiling mode, default = 1
- <b>`storage_order`</b> (int):  storage order, 0 for row major, 1 for column major, default = 0

**Returns:**

- <b>`res`</b> (numpy.ndarray):  a tensor of size (N x InChannels x OutHeight x OutWidth).
- <b>`See https`</b>: //pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1535"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_cast`

```python
numpy_cast(data: ndarray, to: int) → Tuple[ndarray]
```

Execute ONNX cast in Numpy.

This function supports casting to booleans, floats, and double for traced values, converting them accordingly. For raw values (used in constant folding or shape computations), any cast is allowed. See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast

**Args:**

- <b>`data`</b> (numpy.ndarray):  Input encrypted tensor
- <b>`to`</b> (int):  integer value of the onnx.TensorProto DataType enum

**Returns:**

- <b>`result`</b> (numpy.ndarray):  a tensor with the required data type

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1565"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_batchnorm`

```python
numpy_batchnorm(
    x: ndarray,
    scale: ndarray,
    bias: ndarray,
    input_mean: ndarray,
    input_var: ndarray,
    epsilon=1e-05,
    momentum=0.9,
    training_mode=0
) → Tuple[ndarray]
```

Compute the batch normalization of the input tensor.

This can be expressed as:

Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-14

**Args:**

- <b>`x`</b> (numpy.ndarray):  tensor to normalize, dimensions are in the form of (N,C,D1,D2,...,Dn),  where N is the batch size, C is the number of channels.
- <b>`scale`</b> (numpy.ndarray):  scale tensor of shape (C,)
- <b>`bias`</b> (numpy.ndarray):  bias tensor of shape (C,)
- <b>`input_mean`</b> (numpy.ndarray):  mean values to use for each input channel, shape (C,)
- <b>`input_var`</b> (numpy.ndarray):  variance values to use for each input channel, shape (C,)
- <b>`epsilon`</b> (float):  avoids division by zero
- <b>`momentum`</b> (float):  momentum used during training of the mean/variance, not used in inference
- <b>`training_mode`</b> (int):  if the model was exported in training mode this is set to 1, else 0

**Returns:**

- <b>`numpy.ndarray`</b>:  Normalized tenso

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1636"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_flatten`

```python
numpy_flatten(x: ndarray, axis: int = 1) → Tuple[ndarray]
```

Flatten a tensor into a 2d array.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Flatten-13.

**Args:**

- <b>`x`</b> (numpy.ndarray):  tensor to flatten
- <b>`axis`</b> (int):  axis after which all dimensions will be flattened (axis=0 gives a 1D output)

**Returns:**

- <b>`result`</b>:  flattened tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1654"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_or`

```python
numpy_or(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute or in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-7

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1672"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_or_float`

```python
numpy_or_float(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute or in numpy according to ONNX spec and cast outputs to floats.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-7

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_round`

```python
numpy_round(a: ndarray) → Tuple[ndarray]
```

Compute round in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Round-11 Remark that ONNX Round operator is actually a rint, since the number of decimals is forced to be 0

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor whose elements to be rounded.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor with rounded input elements.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1707"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_pow`

```python
numpy_pow(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute pow in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-13

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor whose elements to be raised.
- <b>`b`</b> (numpy.ndarray):  The power to which we want to raise.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1833"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_floor`

```python
numpy_floor(x: ndarray) → Tuple[ndarray]
```

Compute Floor in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Floor-1

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1847"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_max`

```python
numpy_max(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute Max in numpy according to ONNX spec.

Computes the max between the first input and a float constant.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-1

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Constant tensor to compare to the first input

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1864"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_min`

```python
numpy_min(a: ndarray, b: ndarray) → Tuple[ndarray]
```

Compute Min in numpy according to ONNX spec.

Computes the minimum between the first input and a float constant.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-1

**Args:**

- <b>`a`</b> (numpy.ndarray):  Input tensor
- <b>`b`</b> (numpy.ndarray):  Constant tensor to compare to the first input

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1881"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_sign`

```python
numpy_sign(x: ndarray) → Tuple[ndarray]
```

Compute Sign in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sign-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1895"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_neg`

```python
numpy_neg(x: ndarray) → Tuple[ndarray]
```

Compute Negative in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sign-9

**Args:**

- <b>`x`</b> (numpy.ndarray):  Input tensor

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L1909"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_concatenate`

```python
numpy_concatenate(*x: ndarray, axis: int) → Tuple[ndarray]
```

Apply concatenate in numpy according to ONNX spec.

See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#concat-13

**Args:**

- <b>`*x (numpy.ndarray)`</b>:  Input tensors to be concatenated.
- <b>`axis`</b> (int):  Which axis to concat on.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  Output tensor.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L2133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `numpy_unfold`

```python
numpy_unfold(
    x: ndarray,
    kernel_shape: Tuple[int, ],
    pads: Tuple[int, ] = None,
    strides: Tuple[int, ] = None
) → Tuple[ndarray]
```

Compute Unfold using Torch.

Currently supports 2d Unfold with torch semantics. This function is ONNX compatible.

See: https://github.com/onnx/onnx/blob/main/docs/Operators.md

**Args:**

- <b>`x`</b> (numpy.ndarray):  input data (many dtypes are supported). Shape is N x C x H x W for 2d
- <b>`kernel_shape`</b> (Tuple\[int, ...\]):  shape of the kernel. Should have 2 elements for 2d conv
- <b>`pads`</b> (Tuple\[int, ...\]):  padding in ONNX format (begin, end) on each axis
- <b>`strides`</b> (Tuple\[int, ...\]):  stride of the convolution on each axis

**Returns:**

- <b>`res`</b> (numpy.ndarray):  a tensor of size (N x InChannels x OutHeight * OutWidth).
- <b>`See https`</b>: //pytorch.org/docs/stable/generated/torch.nn.Unfold.html

**Raises:**

- <b>`AssertionError`</b>:  if the unfold arguments are wrong

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RawOpOutput`

Type construct that marks an ndarray as a raw output of a quantized op.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ONNXMixedFunction`

A mixed quantized-raw valued onnx function.

ONNX functions will take inputs which can be either quantized or float. Some functions only take quantized inputs, but some functions take both types. For mixed functions we need to tag the parameters that do not need quantization. Thus quantized ops can know which inputs are not QuantizedArray and we avoid unnecessary wrapping of float values as QuantizedArrays.

<a href="../../../src/concrete/ml/onnx/ops_impl.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(function, non_quant_params: Set[str], output_is_raw: bool = False)
```

Create the mixed function and raw parameter list.

**Args:**

- <b>`function`</b> (Any):  function to be decorated
- <b>`non_quant_params`</b>:  Set\[str\]: set of parameters that will not be quantized (stored  as numpy.ndarray)
- <b>`output_is_raw`</b> (bool):  indicates whether the op outputs a value that should  not be quantized
