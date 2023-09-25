<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn`

Import sklearn models.

## **Global Variables**

- **qnn_module**
- **tree_to_numpy**
- **base**
- **glm**
- **linear_model**
- **neighbors**
- **qnn**
- **rf**
- **svm**
- **tree**
- **xgb**

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/__init__.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_models`

```python
get_sklearn_models()
```

Return the list of available models in Concrete ML.

**Returns:**
the lists of models in Concrete ML

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/__init__.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_linear_models`

```python
get_sklearn_linear_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: List[str] = None
)
```

Return the list of available linear models in Concrete ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (List\[str\]):  if not None, only return models with the given string or  list of strings as a substring in their class name

**Returns:**
the lists of linear models in Concrete ML

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/__init__.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_tree_models`

```python
get_sklearn_tree_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: List[str] = None
)
```

Return the list of available tree models in Concrete ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (List\[str\]):  if not None, only return models with the given string or  list of strings as a substring in their class name

**Returns:**
the lists of tree models in Concrete ML

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/__init__.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_neural_net_models`

```python
get_sklearn_neural_net_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: List[str] = None
)
```

Return the list of available neural net models in Concrete ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (List\[str\]):  if not None, only return models with the given string or  list of strings as a substring in their class name

**Returns:**
the lists of neural net models in Concrete ML

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/__init__.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_neighbors_models`

```python
get_sklearn_neighbors_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: List[str] = None
)
```

Return the list of available neighbor models in Concrete ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (List\[str\]):  if not None, only return models with the given string or  list of strings as a substring in their class name

**Returns:**
the lists of neighbor models in Concrete ML
