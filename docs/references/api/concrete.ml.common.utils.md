<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.utils`

Utils that can be re-used by other pieces of code in the module.

## **Global Variables**

- **SUPPORTED_FLOAT_TYPES**
- **SUPPORTED_INT_TYPES**
- **SUPPORTED_TYPES**
- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**
- **USE_OLD_VL**
- **QUANT_ROUND_LIKE_ROUND_PBS**

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `replace_invalid_arg_name_chars`

```python
replace_invalid_arg_name_chars(arg_name: str) → str
```

Sanitize arg_name, replacing invalid chars by \_.

This does not check that the starting character of arg_name is valid.

**Args:**

- <b>`arg_name`</b> (str):  the arg name to sanitize.

**Returns:**

- <b>`str`</b>:  the sanitized arg name, with only chars in \_VALID_ARG_CHARS.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generate_proxy_function`

```python
generate_proxy_function(
    function_to_proxy: Callable,
    desired_functions_arg_names: Iterable[str]
) → Tuple[Callable, Dict[str, str]]
```

Generate a proxy function for a function accepting only \*args type arguments.

This returns a runtime compiled function with the sanitized argument names passed in desired_functions_arg_names as the arguments to the function.

**Args:**

- <b>`function_to_proxy`</b> (Callable):  the function defined like def f(\*args) for which to return a  function like f_proxy(arg_1, arg_2) for any number of arguments.
- <b>`desired_functions_arg_names`</b> (Iterable\[str\]):  the argument names to use, these names are  sanitized and the mapping between the original argument name to the sanitized one is  returned in a dictionary. Only the sanitized names will work for a call to the proxy  function.

**Returns:**

- <b>`Tuple[Callable, Dict[str, str]]`</b>:  the proxy function and the mapping of the original arg name  to the new and sanitized arg names.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_onnx_opset_version`

```python
get_onnx_opset_version(onnx_model: ModelProto) → int
```

Return the ONNX opset_version.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the model.

**Returns:**

- <b>`int`</b>:  the version of the model

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `manage_parameters_for_pbs_errors`

```python
manage_parameters_for_pbs_errors(
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None
)
```

Return (p_error, global_p_error) that we want to give to Concrete.

The returned (p_error, global_p_error) depends on user's parameters and the way we want to manage defaults in Concrete ML, which may be different from the way defaults are managed in Concrete.

Principle:
\- if none are set, we set global_p_error to a default value of our choice
\- if both are set, we raise an error
\- if one is set, we use it and forward it to Concrete

Note that global_p_error is currently set to 0 in the FHE simulation mode.

**Args:**

- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS.
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit.

**Returns:**

- <b>`(p_error, global_p_error)`</b>:  parameters to give to the compiler

**Raises:**

- <b>`ValueError`</b>:  if the two parameters are set (this is _not_ as in Concrete-Python)

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_there_is_no_p_error_options_in_configuration`

```python
check_there_is_no_p_error_options_in_configuration(configuration)
```

Check the user did not set p_error or global_p_error in configuration.

It would be dangerous, since we set them in direct arguments in our calls to Concrete-Python.

**Args:**

- <b>`configuration`</b>:  Configuration object to use  during compilation

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_model_class`

```python
get_model_class(model_class)
```

Return the class of the model (instantiated or not), which can be a partial() instance.

**Args:**

- <b>`model_class`</b>:  The model, which can be a partial() instance.

**Returns:**
The model's class.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_model_class_in_a_list`

```python
is_model_class_in_a_list(model_class, a_list)
```

Indicate if a model class, which can be a partial() instance, is an element of a_list.

**Args:**

- <b>`model_class`</b>:  The model, which can be a partial() instance.
- <b>`a_list`</b>:  The list in which to look into.

**Returns:**
If the model's class is in the list or not.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_model_name`

```python
get_model_name(model_class)
```

Return the name of the model, which can be a partial() instance.

**Args:**

- <b>`model_class`</b>:  The model, which can be a partial() instance.

**Returns:**
the model's name.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_classifier_or_partial_classifier`

```python
is_classifier_or_partial_classifier(model_class)
```

Indicate if the model class represents a classifier.

**Args:**

- <b>`model_class`</b>:  The model class, which can be a functool's `partial` class.

**Returns:**

- <b>`bool`</b>:  If the model class represents a classifier.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_regressor_or_partial_regressor`

```python
is_regressor_or_partial_regressor(model_class)
```

Indicate if the model class represents a regressor.

**Args:**

- <b>`model_class`</b>:  The model class, which can be a functool's `partial` class.

**Returns:**

- <b>`bool`</b>:  If the model class represents a regressor.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_pandas_dataframe`

```python
is_pandas_dataframe(input_container: Any) → bool
```

Indicate if the input container is a Pandas DataFrame.

This function is inspired from Scikit-Learn's test validation tools and avoids the need to add and import Pandas as an additional dependency to the project. See https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/utils/validation.py#L629

**Args:**

- <b>`input_container`</b> (Any):  The input container to consider

**Returns:**

- <b>`bool`</b>:  If the input container is a DataFrame

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_pandas_series`

```python
is_pandas_series(input_container: Any) → bool
```

Indicate if the input container is a Pandas Series.

This function is inspired from Scikit-Learn's test validation tools and avoids the need to add and import Pandas as an additional dependency to the project. See https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/utils/validation.py#L629

**Args:**

- <b>`input_container`</b> (Any):  The input container to consider

**Returns:**

- <b>`bool`</b>:  If the input container is a Series

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_pandas_type`

```python
is_pandas_type(input_container: Any) → bool
```

Indicate if the input container is a Pandas DataFrame or Series.

**Args:**

- <b>`input_container`</b> (Any):  The input container to consider

**Returns:**

- <b>`bool`</b>:  If the input container is a DataFrame orSeries

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_dtype_and_cast`

```python
check_dtype_and_cast(
    values: Any,
    expected_dtype: str,
    error_information: Optional[str] = ''
)
```

Convert any allowed type into an array and cast it if required.

If values types don't match with any supported type or the expected dtype, raise a ValueError.

**Args:**

- <b>`values`</b> (Any):  The values to consider
- <b>`expected_dtype`</b> (str):  The expected dtype, either "float32" or "int64"
- <b>`error_information`</b> (str):  Additional information to put in front of the error message when  raising a ValueError. Default to None.

**Returns:**

- <b>`(Union[numpy.ndarray, torch.utils.data.dataset.Subset])`</b>:  The values with proper dtype.

**Raises:**

- <b>`ValueError`</b>:  If the values' dtype don't match the expected one or casting is not possible.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_bits_precision`

```python
compute_bits_precision(x: ndarray) → int
```

Compute the number of bits required to represent x.

**Args:**

- <b>`x`</b> (numpy.ndarray):  Integer data

**Returns:**

- <b>`int`</b>:  the number of bits required to represent x

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L499"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_brevitas_model`

```python
is_brevitas_model(model: Module) → bool
```

Check if a model is a Brevitas type.

**Args:**

- <b>`model`</b>:  PyTorch model.

**Returns:**

- <b>`bool`</b>:  True if `model` is a Brevitas network.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_tuple`

```python
to_tuple(x: Any) → tuple
```

Make the input a tuple if it is not already the case.

**Args:**

- <b>`x`</b> (Any):  The input to consider. It can already be an input.

**Returns:**

- <b>`tuple`</b>:  The input as a tuple.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L533"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `all_values_are_integers`

```python
all_values_are_integers(*values: Any) → bool
```

Indicate if all unpacked values are of a supported integer dtype.

**Args:**

- <b>`*values (Any)`</b>:  The values to consider.

**Returns:**

- <b>`bool`</b>:  Whether all values are supported integers or not.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `all_values_are_floats`

```python
all_values_are_floats(*values: Any) → bool
```

Indicate if all unpacked values are of a supported float dtype.

**Args:**

- <b>`*values (Any)`</b>:  The values to consider.

**Returns:**

- <b>`bool`</b>:  Whether all values are supported floating points or not.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `all_values_are_of_dtype`

```python
all_values_are_of_dtype(*values: Any, dtypes: Union[str, List[str]]) → bool
```

Indicate if all unpacked values are of the specified dtype(s).

**Args:**

- <b>`*values (Any)`</b>:  The values to consider.
- <b>`dtypes`</b> (Union\[str, List\[str\]\]):  The dtype(s) to consider.

**Returns:**

- <b>`bool`</b>:  Whether all values are of the specified dtype(s) or not.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L587"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `array_allclose_and_same_shape`

```python
array_allclose_and_same_shape(
    a,
    b,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False
) → bool
```

Check if two numpy arrays are equal within a tolerances and have the same shape.

**Args:**

- <b>`a`</b> (numpy.ndarray):  The first input array
- <b>`b`</b> (numpy.ndarray):  The second input array
- <b>`rtol`</b> (float):  The relative tolerance parameter
- <b>`atol`</b> (float):  The absolute tolerance parameter
- <b>`equal_nan`</b> (bool):  Whether to compare NaN’s as equal. If True, NaN’s in a will be considered  equal to NaN’s in b in the output array

**Returns:**

- <b>`bool`</b>:  True if the arrays have the same shape and all elements are equal within the specified  tolerances, False otherwise.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `process_rounding_threshold_bits`

```python
process_rounding_threshold_bits(rounding_threshold_bits)
```

Check and process the rounding_threshold_bits parameter.

**Args:**

- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  Defines precision  rounding for model accumulators. Accepts None, an int, or a dict.  The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)  and 'n_bits' ('auto' or int)

**Returns:**

- <b>`Dict[str, Union[str, int]]`</b>:  Processed rounding_threshold_bits dictionary.

**Raises:**

- <b>`NotImplementedError`</b>:  If 'auto' rounding is specified but not implemented.
- <b>`ValueError`</b>:  If an invalid type or value is provided for rounding_threshold_bits.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/utils.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FheMode`

Enum representing the execution mode.

This enum inherits from str in order to be able to easily compare a string parameter to its equivalent Enum attribute.

**Examples:**
fhe_disable = FheMode.DISABLE

` fhe_disable == "disable"`
True

```
 >>> fhe_disable == "execute"
 False

 >>> FheMode.is_valid("simulate")
 True

 >>> FheMode.is_valid(FheMode.EXECUTE)
 True

 >>> FheMode.is_valid("predict_in_fhe")
 False
```
