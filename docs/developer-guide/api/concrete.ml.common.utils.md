<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.utils`

Utils that can be re-used by other pieces of code in the module.

## **Global Variables**

- **DEFAULT_P_ERROR_PBS**

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
