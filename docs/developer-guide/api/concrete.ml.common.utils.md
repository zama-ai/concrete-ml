<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.utils`

Utils that can be re-used by other pieces of code in the module.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `manage_parameters_for_pbs_errors`

```python
manage_parameters_for_pbs_errors(
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None
)
```

Return (p_error, global_p_error) that we want to give to Concrete-Numpy and the compiler.

The returned (p_error, global_p_error) depends on user's parameters and the way we want to manage defaults in Concrete-ML, which may be different from the way defaults are managed in Concrete-Numpy

Principle:
\- if none are set, we set global_p_error to a default value of our choice
\- if both are set, we raise an error
\- if one is set, we use it and forward it to Concrete-Numpy and the compiler

Note that global_p_error is currently not simulated by the VL, i.e., taken as 0.

**Args:**

- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS.
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit.

**Returns:**

- <b>`(p_error, global_p_error)`</b>:  parameters to give to the compiler

**Raises:**

- <b>`ValueError`</b>:  if the two parameters are set (this is _not_ as in Concrete-Numpy)

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_there_is_no_p_error_options_in_configuration`

```python
check_there_is_no_p_error_options_in_configuration(configuration)
```

Check the user did not set p_error or global_p_error in configuration.

It would be dangerous, since we set them in direct arguments in our calls to Concrete-Numpy.

**Args:**

- <b>`configuration`</b>:  Configuration object to use  during compilation

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_model_class_in_a_list`

```python
is_model_class_in_a_list(model_class, a_list)
```

Say if model_class (which may be a partial()) is an element of a_list.

**Args:**

- <b>`model_class`</b>:  the model
- <b>`a_list`</b>:  the list in which to look

**Returns:**
whether the class is in the list

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_model_name`

```python
get_model_name(model_class)
```

Return a model (which may be a partial()) name.

**Args:**

- <b>`model_class`</b>:  the model

**Returns:**
the class name
