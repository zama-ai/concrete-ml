# Prediction with FHE

Concrete ML has APIs that make it easy, during model development and testing, to perform encryption, execution in FHE, and decryption in a single step. For more control, these individual steps can be executed separately. The APIs used to accomplish this are different for:

- [Built-in models](#built-in-models)
- [Custom models](#custom-models)

## Built-in models

The following example shows how to create a synthetic data-set and how to use it to train a LogisticRegression model from Concrete ML.
Next, we will discuss the dedicated functions for encryption, inference, and decryption.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression
import numpy

# Create a synthetic data-set for a classification problem
x, y = make_classification(n_samples=100, class_sep=2, n_features=3, n_informative=3, n_redundant=0, random_state=42)

# Split the data-set into a train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instantiate and train the model
model = LogisticRegression()
model.fit(x_train,y_train)

# Simulate the predictions in the clear (optional)
y_pred_clear = model.predict(x_test)

# Compile the model on a representative set
fhe_circuit = model.compile(x_train)
```

All Concrete ML built-in models have a monolithic `predict` method that performs the encryption, FHE execution, and decryption with a single function call. Concrete ML models follow the same API as scikit-learn models, transparently performing the steps related to encryption for convenience.

<!--pytest-codeblocks:cont-->

```python
# Predict in FHE
y_pred_fhe = model.predict(x_test, fhe="execute")
```

Regarding this LogisticRegression model, as with scikit-learn, it is possible to predict the logits as well as the class probabilities by respectively using the `decision_function` or `predict_proba` methods instead.

Alternatively, it is possible to execute all main steps (key generation, quantization, encryption, FHE execution, decryption) separately.

<!--pytest-codeblocks:cont-->

```python
# Generate the keys (set force to True in order to generate new keys at each execution)
fhe_circuit.keygen(force=True)

y_pred_fhe_step = []

for f_input in x_test:
    # Quantize an input (float)
    q_input = model.quantize_input([f_input])
    
    # Encrypt the input
    q_input_enc = fhe_circuit.encrypt(q_input)

    # Execute the linear product in FHE 
    q_y_enc = fhe_circuit.run(q_input_enc)

    # Decrypt the result (integer)
    q_y = fhe_circuit.decrypt(q_y_enc)

    # De-quantize the result
    y = model.dequantize_output(q_y)

    # Apply either the sigmoid if it is a binary classification task, which is the case in this 
    # example, or a softmax function in order to get the probabilities (in the clear)
    y_proba = model.post_processing(y)

    # Since this model does classification, apply the argmax to get the class predictions (in the clear)
    # Note that regression models won't need the following line
    y_class = numpy.argmax(y_proba, axis=1)

    y_pred_fhe_step += list(y_class)

y_pred_fhe_step = numpy.array(y_pred_fhe_step)

print("Predictions in clear:", y_pred_clear)
print("Predictions in FHE  :", y_pred_fhe_step)
print(f"Similarity: {int((y_pred_fhe_step == y_pred_clear).mean()*100)}%")
```

## Custom models

For custom models, the API to execute inference in FHE or simulation is illustrated as:

<!--pytest-codeblocks:cont-->

```python
from torch import nn
from brevitas import nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model

class FCSmall(nn.Module):
    """A small QAT NN."""

    def __init__(self, input_output):
        super().__init__()
        self.quant_input = qnn.QuantIdentity(bit_width=3)
        self.fc1 = qnn.QuantLinear(in_features=input_output, out_features=input_output, weight_bit_width=3, bias=True)
        self.act_f = nn.ReLU()
        self.fc2 = qnn.QuantLinear(in_features=input_output, out_features=input_output, weight_bit_width=3, bias=True)

    def forward(self, x):
        return self.fc2(self.act_f(self.fc1(self.quant_input(x))))

torch_model = FCSmall(3)

quantized_module = compile_brevitas_qat_model(
    torch_model,
    x_train,
)

x_test_q = quantized_module.quantize_input(x_test)
y_pred = quantized_module.quantized_forward(x_test_q, fhe="simulate")
y_pred = quantized_module.dequantize_output(y_pred)

y_pred = numpy.argmax(y_pred, axis=1)
```
