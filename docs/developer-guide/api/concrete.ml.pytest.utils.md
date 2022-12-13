<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/pytest/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pytest.utils`

Common functions or lists for test files, which can't be put in fixtures.

## **Global Variables**

- **regressor_models**
- **classifier_models**
- **classifiers**
- **regressors**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/pytest/utils.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sanitize_test_and_train_datasets`

```python
sanitize_test_and_train_datasets(model, x, y)
```

Sanitize datasets depending on the model type.

**Args:**

- <b>`model`</b>:  the model
- <b>`x`</b>:  the first output of load_data, i.e., the inputs
- <b>`y`</b>:  the second output of load_data, i.e., the labels

**Returns:**
Tuple containing sanitized (model_params, x, y, x_train, y_train, x_test)
