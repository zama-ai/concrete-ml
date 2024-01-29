<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.fhe_client_server`

APIs for FHE deployment.

## **Global Variables**

- **CML_VERSION**

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_concrete_versions`

```python
check_concrete_versions(zip_path: Path)
```

Check that current versions match the ones used in development.

This function loads the version JSON file found in client.zip or server.zip files and then checks that current package versions (Concrete Python, Concrete ML) as well as the Python current version all match the ones that are currently installed.

**Args:**

- <b>`zip_path`</b> (Path):  The path to the client or server zip file that contains the version.json  file to check.

**Raises:**

- <b>`ValueError`</b>:  If at least one version mismatch is found.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelServer`

Server API to load and run the FHE circuit.

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load()
```

Load the circuit.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelDev`

Dev API to save the model and then load and run the FHE circuit.

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str, model: Any = None)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved
- <b>`model`</b> (Any):  the model to use for the FHE API

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(via_mlir: bool = False)
```

Export all needed artifacts for the client and server.

**Arguments:**

- <b>`via_mlir`</b> (bool):  serialize with `via_mlir` option from Concrete-Python.  For more details on the topic please refer to Concrete-Python's documentation.

**Raises:**

- <b>`Exception`</b>:  path_dir is not empty

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelClient`

Client API to encrypt and decrypt FHE data.

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str, key_dir: Optional[str] = None)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved
- <b>`key_dir`</b> (str):  the path to the directory where the keys are stored

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize_decrypt_dequantize`

```python
deserialize_decrypt_dequantize(
    serialized_encrypted_quantized_result: bytes
) → ndarray
```

Deserialize, decrypt and de-quantize the values.

**Args:**

- <b>`serialized_encrypted_quantized_result`</b> (bytes):  the serialized, encrypted  and quantized result

**Returns:**

- <b>`numpy.ndarray`</b>:  the decrypted (de-quantized) values

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_private_and_evaluation_keys`

```python
generate_private_and_evaluation_keys(force=False)
```

Generate the private and evaluation keys.

**Args:**

- <b>`force`</b> (bool):  if True, regenerate the keys even if they already exist

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L291"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_serialized_evaluation_keys`

```python
get_serialized_evaluation_keys() → bytes
```

Get the serialized evaluation keys.

**Returns:**

- <b>`bytes`</b>:  the evaluation keys

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load()
```

Load the quantizers along with the FHE specs.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_encrypt_serialize`

```python
quantize_encrypt_serialize(x: ndarray) → bytes
```

Quantize, encrypt and serialize the values.

**Args:**

- <b>`x`</b> (numpy.ndarray):  the values to quantize, encrypt and serialize

**Returns:**

- <b>`bytes`</b>:  the quantized, encrypted and serialized values
