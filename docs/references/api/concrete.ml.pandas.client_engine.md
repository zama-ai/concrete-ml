<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/pandas/client_engine.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pandas.client_engine`

Define the framework used for managing keys (encrypt, decrypt) for encrypted data-frames.

## **Global Variables**

- **CURRENT_API_VERSION**

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/client_engine.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClientEngine`

Define a framework that manages keys.

<a href="../../../src/concrete/ml/pandas/client_engine.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(keygen: bool = True, keys_path: Optional[Path, str] = None)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/client_engine.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt_to_pandas`

```python
decrypt_to_pandas(encrypted_dataframe: EncryptedDataFrame) → DataFrame
```

Decrypt an encrypted data-frame using the loaded client and return a Pandas data-frame.

**Args:**

- <b>`encrypted_dataframe`</b> (EncryptedDataFrame):  The encrypted data-frame to decrypt.

**Returns:**

- <b>`pandas.DataFrame`</b>:  The Pandas data-frame built on the decrypted values.

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/client_engine.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt_from_pandas`

```python
encrypt_from_pandas(pandas_dataframe: DataFrame) → EncryptedDataFrame
```

Encrypt a Pandas data-frame using the loaded client.

**Args:**

- <b>`pandas_dataframe`</b> (DataFrame):  The Pandas data-frame to encrypt.

**Returns:**

- <b>`EncryptedDataFrame`</b>:  The encrypted data-frame.

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/client_engine.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen`

```python
keygen(keys_path: Optional[Path, str] = None)
```

Generate the keys.

**Args:**

- <b>`keys_path`</b> (Optional\[Union\[Path, str\]\]):  The path where to save the keys. Note that if  some keys already exist in that path, the client will use them instead of generating  new ones. Default to None.
