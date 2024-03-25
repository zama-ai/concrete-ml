# Working with Encrypted DataFrames

Concrete ML extends pandas functionality by introducing the capability to perform operations on encrypted data using Fully Homomorphic Encryption (FHE). This API ensures data scientists can leverage well-known pandas-like operations while maintaining privacy throughout the whole process.

Potential applications include:
- Conducting data analysis securely on data stored remotely
- Joint data analysis efforts between multiple parties
- Data preparation steps before machine learning tasks, such as inference or training


## Encrypt and Decrypt a DataFrame

To encrypt a pandas DataFrame, the `encrypt_from_pandas` function is used. A client object must be obtained using `load_client`.

<!--pytest-codeblocks:skip-->
```python
from concrete.ml.pandas import encrypt_from_pandas, load_client

# Load your pandas DataFrame
df = pandas.read_csv("path_to_your_data.csv")

# Obtain client object
client = load_client()

# Encrypt the DataFrame
df_encrypted = encrypt_from_pandas(df, client)

# Decrypt the DataFrame
df_decrypted = df_encrypted.decrypt_to_pandas(client)
```

## Supported Operations on Encrypted DataFrames

Encrypted DataFrames support a subset of operations that are available for pandas DataFrames. The following operations are currently supported:


<!--pytest-codeblocks:skip-->	
```python
df_encrypted_merged = df_encrypted1.merge(df_encrypted2, how="left", on="common_column")
```


> **Important**: The merge operation on Encrypted DataFrames can be **securely** performed on a third-party server. This means that the server can execute the merge without ever having access to the unencrypted data. The server only requires the encrypted DataFrames.


## Serialization of Encrypted DataFrames

Encrypted DataFrames can be serialized to a file format for storage or transfer.


### Saving DataFrames

<!--pytest-codeblocks:skip-->	
```python
df_encrypted_merged.save("df_encrypted_merged")
```

### Loading DataFrames

To load an encrypted DataFrame from a file:

<!--pytest-codeblocks:skip-->	
```python
from concrete.ml.pandas import load_encrypted_dataframe
df_encrypted_merged = load_encrypted_dataframe("df_encrypted_merged")
```
## Supported Data Types and Schema Definition

Concrete ML's encrypted DataFrame operations support a specific set of data types:

- **Integer**:  Integers are supported within a specific range determined by the encryption scheme's quantization parameters. Default range is 1 to 15. 0 being used for the `NaN`. Values outside this range will cause a `ValueError` to be raised during the pre-processing stage.
- **Quantized Float**: Floating-point numbers are quantized to integers within the supported range. This is achieved by computing a scale and zero point for each column, which are used to map the floating-point numbers to the quantized integer space.
- **String Enum**: String columns are mapped to integers starting from 1. This mapping is stored and later used for dequantization. If the number of unique strings exceeds 15, a `ValueError` is raised.

## Error Handling

The library is designed to raise specific errors when encountering issues during the pre-processing and post-processing stages:
- `ValueError`: Raised when a column contains values outside the allowed range for integers, when there are too many unique strings, or when encountering an unsupported data type.
- `TypeError`: Raised when an operation is attempted on a data type that is not supported by the operation.


## Example Workflow

An example workflow where two clients encrypt their respective DataFrames, perform a merge operation on the server side, and then decrypt the results is available in the notebook [encrypted_pandas.ipynb](concrete-ml/use_case_examples/dataframe/encrypted_pandas.ipynb).


## Current Limitations

While this API offers a new secure way to work on remotely stored and encrypted data, it's important to recognize that it is still in its early stages.

Current Limitations:

- **Precision of Values**: The precision for numerical values is limited to 4 bits.
- **Supported Operations**: The `merge` operation is the only one available for encrypted DataFrames.
- **Index Handling**: Index values are not preserved; users should move any relevant data from the index to a dedicated new column before encrypting.
- **Integer Range**: The range of integers that can be encrypted is between 1 and 15.
- **Uniqueness for `merge`**: The `merge` operation requires that the columns to merge on contain unique values.
- **Metadata Security**: Column names and the mapping of strings to integers are not encrypted and are sent to the server in clear text.
