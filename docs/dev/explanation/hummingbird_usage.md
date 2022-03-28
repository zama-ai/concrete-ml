# HummingBird Usage

## Why Hummingbird?

Hummingbird contains an interesting feature for **Concrete ML**: it converts many algorithms (see [supported algorithms](https://microsoft.github.io/hummingbird/api/hummingbird.ml.supported.html)) to tensor computations using a specific backend (torch, TorchScript, ONNX and TVM).

**Concrete ML** allows to convert an ONNX inference to numpy inference (note that numpy is always our entry point to run models in FHE).

## Usage

We use a simple functionnality of Hummingbird which is the `convert` function that can be imported as follows from the `hummingbird.ml` package:

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

In theory, we can directly use this `onnx_model` within our `get_equivalent_numpy_forward` (as long as all operators present in the ONNX are implemented in numpy) and get the numpy inference.

In practice, we have some steps to clean the ONNX and make the graph compatible with our framework such as:

- applying quantization where needed
- deleting non FHE friendly ONNX operators such as *Softmax* and *ArgMax*.
