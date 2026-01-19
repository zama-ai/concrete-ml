# Logistic Regression with FHE

```python
from concrete.ml.sklearn import LogisticRegression

# Обучение модели на открытых данных
clf = LogisticRegression(n_bits=3)
clf.fit(X_train, y_train)

# Шифрование входных данных и предсказание
cipher_x = clf.encrypt(X_test[0])
encrypted_pred = clf.predict(cipher_x)
decrypted_pred = encrypted_pred.decrypt()
print("Decrypted prediction:", decrypted_pred)
