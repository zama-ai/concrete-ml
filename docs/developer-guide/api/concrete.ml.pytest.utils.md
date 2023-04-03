<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/pytest/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pytest.utils`

Common functions or lists for test files, which can't be put in fixtures.

## **Global Variables**

- **sklearn_models_and_datasets**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/pytest/utils.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_random_extract_of_sklearn_models_and_datasets`

```python
get_random_extract_of_sklearn_models_and_datasets()
```

Return a random sublist of sklearn_models_and_datasets.

The sublist contains exactly one model of each kind.

**Returns:**
the sublist

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/pytest/utils.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `instantiate_model_generic`

```python
instantiate_model_generic(model_class, **parameters)
```

Instantiate any Concrete-ML model type.

**Args:**

- <b>`model_class`</b> (class):  The type of the model to instantiate
- <b>`parameters`</b> (dict):  Hyper-parameters for the model instantiation

**Returns:**

- <b>`model_name`</b> (str):  The type of the model as a string
- <b>`model`</b> (object):  The model instance

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/pytest/utils.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_torchvision_dataset`

```python
get_torchvision_dataset(param: Dict, train_set: bool)
```

Get train or testing data-set.

**Args:**

- <b>`param`</b> (Dict):  Set of hyper-parameters to use based on the selected torchvision data-set.
- <b>`It must contain`</b>:  data-set transformations (torchvision.transforms.Compose), and the data-set_size (Optional\[int\]).
- <b>`train_set`</b> (bool):  Use train data-set if True, else testing data-set

**Returns:**
A torchvision datasets.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/pytest/utils.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `data_calibration_processing`

```python
data_calibration_processing(data, n_sample: int, targets=None)
```

Reduce size of the given dataset.

**Args:**

- <b>`data`</b>:  The input container to consider
- <b>`n_sample`</b> (int):  Number of samples to keep if the given data-set
- <b>`targets`</b>:  If `dataset` is a `torch.utils.data.Dataset`, it typically contains both the data  and the corresponding targets. In this case, `targets` must be set to `None`.  If `data` is instance of `torch.Tensor` or 'numpy.ndarray`, `targets\` is expected.

**Returns:**

- <b>`Tuple[numpy.ndarray, numpy.ndarray]`</b>:  The input data and the target (respectively x and y).

**Raises:**

- <b>`TypeError`</b>:  If the 'data-set' does not match any expected type.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/release/1.0.x/src/concrete/ml/pytest/utils.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_torch_model`

```python
load_torch_model(
    model_class: Module,
    state_dict_or_path: Optional[str, Path, Dict[str, Any]],
    params: Dict,
    device: str = 'cpu'
) → Module
```

Load an object saved with torch.save() from a file or dict.

**Args:**

- <b>`model_class`</b> (torch.nn.Module):  A Pytorch or Brevitas network.
- <b>`state_dict_or_path`</b> (Optional\[Union\[str, Path, Dict\[str, Any\]\]\]):  Path or state_dict
- <b>`params`</b> (Dict):  Model's parameters
- <b>`device`</b> (str):   Device type.

**Returns:**

- <b>`torch.nn.Module`</b>:  A Pytorch or Brevitas network.
