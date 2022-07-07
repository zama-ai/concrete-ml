# Using HummingBird

[Hummingbird](https://microsoft.github.io/hummingbird/) is a third party open-source library that converts machine learning models into tensor computations. Many algorithms (see [supported algorithms](https://microsoft.github.io/hummingbird/api/hummingbird.ml.supported.html)) are converted using a specific backend (torch, torchscript, ONNX and TVM).

Concrete-ML allows the conversion of an ONNX inference to NumPy inference (note that NumPy is always the entry point to run models in FHE with Concrete ML).

## Usage

Hummingbird exposes a `convert` function that can be imported as follows from the `hummingbird.ml` package:

```python
# Disable Hummingbird warnings for pytest.
import warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert
```

This function can be used to convert a machine learning model to an ONNX as follows:

<!--pytest-codeblocks:cont-->

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Instantiate the logistic regression from sklearn
lr = LogisticRegression()

# Create synthetic data
X, y = make_classification(
    n_samples=100, n_features=20, n_classes=2
)

# Fit the model
lr.fit(X, y)

# Convert the model to ONNX
onnx_model = convert(lr, backend="onnx", test_input=X).model
```

In theory, the resulting `onnx_model` could be used directly within Concrete-ML's `get_equivalent_numpy_forward` method (as long as all operators present in the ONNX model are implemented in NumPy) and get the NumPy inference.

In practice, there are some steps needed to clean the ONNX output and make the graph compatible with Concrete-ML, such as applying quantization where needed or deleting/replacing non-FHE friendly ONNX operators (such as _Softmax_ and _ArgMax)._

















































































































































































































































