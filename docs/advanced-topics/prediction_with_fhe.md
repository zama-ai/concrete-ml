# Prediction with FHE

Concrete-ML offers the possibility to easily encrypt the inputs with a dedicated function, to execute
the inference on these inputs encrypted, and, finally, to decrypt the outputs with another dedicated function.

The following example shows how to create a synthetic data-set and how to use it
to train a LogisticRegression model from Concrete-ML.
Next, the dedicated functions for encryption, inference and decryption are discussed.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression
import numpy

# Create a synthetic data-set for a classification problem
x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)

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

All Concrete-ML built-in models have a monolithic `predict` method that performs the encryption, FHE execution and decryption with a single function call.
Thus, Concrete-ML models follow the same API as Scikit-Learn models, transparently performing the steps related to encryption to accentuate convenience.

<!--pytest-codeblocks:cont-->

```python
# Predict in FHE
y_pred_fhe = model.predict(x_test, fhe="execute")
```

Regarding this LogisticRegression model, as with Scikit-Learn, it is possible to predict the logits as well as the class probabilities by respectively using the `decision_function` or `predict_proba` methods instead.

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

    # Dequantize the result
    y = model.dequantize_output(q_y)

    # Apply either the sigmoid if it is a binary classification task, which is the case in this 
    # example, or a softmax function in order to get the probabilities (in the clear)
    y_proba = model.post_processing(y)

    # Apply the argmax to get the class predictions (in the clear)
    y_class = numpy.argmax(y_proba, axis=1)

    y_pred_fhe_step += list(y_class)

y_pred_fhe_step = numpy.array(y_pred_fhe_step)

print("Predictions in clear:", y_pred_clear)
print("Predictions in FHE  :", y_pred_fhe_step)
print(f"Similarity: {int((y_pred_fhe_step == y_pred_clear).mean()*100)}%")
```
