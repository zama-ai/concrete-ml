<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.fhe_client_server`

APIs for FHE deployment.

## **Global Variables**

- **CML_VERSION**

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DeploymentMode`

Mode for the FHE API.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelServer`

Server API to load and run the FHE circuit.

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load()
```

Load the circuit.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(
    serialized_encrypted_quantized_data: Union[bytes, Value, Tuple[bytes, ], Tuple[Value, ]],
    serialized_evaluation_keys: bytes
) → Union[bytes, Value, Tuple[bytes, ], Tuple[Value, ]]
```

Run the model on the server over encrypted data.

**Args:**

- <b>`serialized_encrypted_quantized_data`</b> (Union\[bytes, fhe.Value, Tuple\[bytes, ...\],                 Tuple\[fhe.Value, ...\]\]):  The encrypted and quantized values to consider. If these  values are serialized (in bytes), they are first deserialized.
- <b>`serialized_evaluation_keys`</b> (bytes):  The evaluation keys. If they are serialized (in  bytes), they are first deserialized.

**Returns:**

- <b>`Union[bytes, fhe.Value, Tuple[bytes, ...], Tuple[fhe.Value, ...]]`</b>:  The model's encrypted  and quantized results. If the inputs were initially serialized, the outputs are also  serialized.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelDev`

Dev API to save the model and then load and run the FHE circuit.

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str, model: Any = None)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved
- <b>`model`</b> (Any):  the model to use for the FHE API

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(
    mode: DeploymentMode = <DeploymentMode.INFERENCE: 'inference'>,
    via_mlir: bool = True
)
```

Export all needed artifacts for the client and server.

**Arguments:**

- <b>`mode`</b> (DeploymentMode):  the mode to save the FHE circuit,  either "inference" or "training".
- <b>`via_mlir`</b> (bool):  serialize with `via_mlir` option from Concrete-Python.

**Raises:**

- <b>`Exception`</b>:  path_dir is not empty or training module does not exist
- <b>`ValueError`</b>:  if mode is not "inference" or "training"

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L362"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHEModelClient`

Client API to encrypt and decrypt FHE data.

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L370"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_dir: str, key_dir: Optional[str] = None)
```

Initialize the FHE API.

**Args:**

- <b>`path_dir`</b> (str):  the path to the directory where the circuit is saved
- <b>`key_dir`</b> (str):  the path to the directory where the keys are stored

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize_decrypt`

```python
deserialize_decrypt(
    *serialized_encrypted_quantized_result: Optional[bytes]
) → Union[Any, Tuple[Any, ]]
```

Deserialize and decrypt the values.

**Args:**

- <b>`serialized_encrypted_quantized_result`</b> (Optional\[bytes\]):  The serialized, encrypted and  quantized values.

**Returns:**

- <b>`Union[Any, Tuple[Any, ...]]`</b>:  The decrypted and deserialized values.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize_decrypt_dequantize`

```python
deserialize_decrypt_dequantize(
    *serialized_encrypted_quantized_result: Optional[bytes]
) → Union[ndarray, Tuple[ndarray, ]]
```

Deserialize, decrypt and de-quantize the values.

**Args:**

- <b>`serialized_encrypted_quantized_result`</b> (Optional\[bytes\]):  The serialized, encrypted and  quantized result

**Returns:**

- <b>`Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]`</b>:  The clear float values.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_private_and_evaluation_keys`

```python
generate_private_and_evaluation_keys(force=False)
```

Generate the private and evaluation keys.

**Args:**

- <b>`force`</b> (bool):  if True, regenerate the keys even if they already exist

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_serialized_evaluation_keys`

```python
get_serialized_evaluation_keys(
    include_tfhers_key: bool = False
) → Union[bytes, Tuple[bytes, Union[bytes, NoneType]]]
```

Get the serialized evaluation keys.

**Args:**

- <b>`include_tfhers_key`</b> (bool):  Whether to include the serialized TFHE-rs public key.

**Returns:**

- <b>`bytes`</b>:  the evaluation keys

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L466"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_tfhers_secret_key`

```python
get_tfhers_secret_key() → Union[bytes, NoneType]
```

Get the secret key that decrypts TFHE-rs ciphertexts.

This secret key use used by the client object to encrypt TFHE-rs ciphertexts that are sent to the server, but also to decrypt outputs produced by a server-side computation, that can be a Concrete ML model or a TFHE-rs program.

**Returns:**

- <b>`bytes`</b>:  the serialized TFHE-rs secret key

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L388"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load()
```

Load the quantizers along with the FHE specs.

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/fhe_client_server.py#L478"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_encrypt_serialize`

```python
quantize_encrypt_serialize(
    *x: Optional[ndarray]
) → Union[bytes, NoneType, Tuple[Union[bytes, NoneType], ]]
```

Quantize, encrypt and serialize the values.

**Args:**

- <b>`x`</b> (Optional\[numpy.ndarray\]):  The values to quantize, encrypt and serialize.

**Returns:**

- <b>`Union[bytes, Tuple[bytes, ...]]`</b>:  The quantized, encrypted and serialized values.
