# Converting a simple model to FHE

**Concrete-ML** makes it easy to convert ML pipelines developed with other frameworks to FHE.
This example shows how to use **Concrete-ML** to train, convert and run a simple classifier in FHE.
The pipeline shown here is inspired by scikit-learn logistic regression.

## Import packages

First, we import useful tools from scikit-learn.

```python
import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

Second, we import `LogisticRegression`, from **Concrete-ML**, which shares the same API as the equivalent scikit-learn class. Indeed, behind the scenes, **Concrete-ML** uses scikit-learn to train this classifier.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.sklearn import LogisticRegression
```

## Data

We now create a synthetic dataset.

<!--pytest-codeblocks:cont-->

```python
N_EXAMPLE_TOTAL = 100
N_TEST = 20
x, y = make_classification(n_samples=N_EXAMPLE_TOTAL,
    class_sep=2, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=N_TEST / N_EXAMPLE_TOTAL, random_state=42
)
```

## Quantization

**Concrete-ML** requires that data and model parameters be integer values, so models trained in floating point need to be quantized, as described in the [the quantization documentation](quantization.md). For this dataset, which has 4 dimensions, we can quantize inputs and parameters to 3 bits.

<!--pytest-codeblocks:cont-->

```python
model = LogisticRegression(n_bits=3)
model.fit(X_train, y_train)
```

## Check quantized accuracy

Once trained and quantized, we can check the accuracy that the model can obtain. Note, that this does not yet use Fully Homomorphic Encryption, as this step is done only to evaluate the model during its development.

<!--pytest-codeblocks:cont-->

```python
y_pred_clear = model.predict(X_test, execute_in_fhe=False)
print(f"Accuracy clear: {numpy.mean(y_pred_clear == y_test)*100}%")
```

## Compile to FHE

We compile the model, to get its equivalent FHE counterpart. We provide a representative calibration set, in this case the same data set as for training:

<!--pytest-codeblocks:cont-->

```python
model.compile(x)
```

## Run the model in FHE

Finally we perform inference in FHE.

<!--pytest-codeblocks:cont-->

```python
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)
```

If you run this, you should see that the inferences in clear and in FHE are equal.

<!--pytest-codeblocks:cont-->

```python
print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print("Comparison:", (y_pred_fhe == y_pred_clear))

# Output:
#   In clear  : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   In FHE    : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   Comparison: [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True]
```

## Setup a client/server protocol

**Concrete ML** provides functionality to deploy FHE machine learning models in a client/server setting. The deployment workflow and model serving follows the following pattern:

### Deployment

The training of the model and its compilation to FHE are performed on a development machine. The compiled model, public cryptographic parameters and model meta data are deployed on the server.

#### Serving

The client obtains the cryptographic parameters and generates private and evaluation keys. Evaluation keys are sent to the server. Then the private data are encrypted by the client and sent to the server. The (FHE) model inference is done on the server with the evaluation keys, on those encrypted private data. The encrypted result is returned by the server to the client, which decrypts it using her private key. The client performs any necessary post-processing of the decrypted result.

#### Example notebook

We refer the reader to [this notebook / file](simple_compilation.md) for a complete description.

```{warning}
FIXME: Jordan, fix the previous link, link to your notebook or markdown
```
