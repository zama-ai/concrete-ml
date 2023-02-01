<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.fhe_client_server`

APIs for FHE deployment.

## **Global Variables**

- **CML_VERSION**
- **AVAILABLE_MODEL**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelServer`

Server API to load and run the FHE circuit.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load()
```

Load the circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(
    serialized_encrypted_quantized_data: bytes,
    serialized_evaluation_keys: bytes
) → bytes
```

Run the model on the server over encrypted data.

**Args:**

- <b>`serialized_encrypted_quantized_data`</b> (bytes):  the encrypted, quantized  and serialized data
- <b>`serialized_evaluation_keys`</b> (bytes):  the serialized evaluation keys

**Returns:**

- <b>`bytes`</b>:  the result of the model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelDev`

Dev API to save the model and then load and run the FHE circuit.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str, model: Any = None)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved
- <b>`model`</b> (Any):  the model to use for the FHE API

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save()
```

Export all needed artifacts for the client and server.

**Raises:**

- <b>`Exception`</b>:  path_dir is not empty

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelClient`

Client API to encrypt and decrypt FHE data.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str, key_dir: Optional[str] = None)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved
- <b>`key_dir`</b> (str):  the path to the directory where the keys are stored

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize_decrypt`

```python
deserialize_decrypt(serialized_encrypted_quantized_result: bytes) → ndarray
```

Deserialize and decrypt the values.

**Args:**

- <b>`serialized_encrypted_quantized_result`</b> (bytes):  the serialized, encrypted  and quantized result

**Returns:**

- <b>`numpy.ndarray`</b>:  the decrypted and deserialized values

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L337"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize_decrypt_dequantize`

```python
deserialize_decrypt_dequantize(
    serialized_encrypted_quantized_result: bytes
) → ndarray
```

Deserialize, decrypt and dequantize the values.

**Args:**

- <b>`serialized_encrypted_quantized_result`</b> (bytes):  the serialized, encrypted  and quantized result

**Returns:**

- <b>`numpy.ndarray`</b>:  the decrypted (dequantized) values

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_private_and_evaluation_keys`

```python
generate_private_and_evaluation_keys(force=False)
```

Generate the private and evaluation keys.

**Args:**

- <b>`force`</b> (bool):  if True, regenerate the keys even if they already exist

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_serialized_evaluation_keys`

```python
get_serialized_evaluation_keys() → bytes
```

Get the serialized evaluation keys.

**Returns:**

- <b>`bytes`</b>:  the evaluation keys

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load()
```

Load the quantizers along with the FHE specs.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/deployment/fhe_client_server.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_encrypt_serialize`

```python
quantize_encrypt_serialize(x: ndarray) → bytes
```

Quantize, encrypt and serialize the values.

**Args:**

- <b>`x`</b> (numpy.ndarray):  the values to quantize, encrypt and serialize

**Returns:**

- <b>`bytes`</b>:  the quantized, encrypted and serialized values
