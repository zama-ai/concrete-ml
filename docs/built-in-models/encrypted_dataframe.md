# Working with Encrypted DataFrames

Concrete ML builds upon the pandas data-frame functionality by introducing the capability to construct and perform operations on encrypted data-frames using FHE. This API ensures data scientists can leverage well-known pandas-like operations while maintaining privacy throughout the whole process.

Encrypted data-frames are a storage format for encrypted tabular data and they can be exchanged with third-parties without security risks.

Potential applications include:

- Encrypted storage of tabular datasets
- Joint data analysis efforts between multiple parties
- Data preparation steps before machine learning tasks, such as inference or training
- Secure outsourcing of data analysis to untrusted third parties

## Encrypt and Decrypt a DataFrame

To encrypt a pandas `DataFrame`, you must construct a `ClientEngine` which manages keys. Then call the `encrypt_from_pandas` function:

```python
from concrete.ml.pandas import ClientEngine
from io import StringIO
import pandas

data_left = """index,total_bill,tip,sex,smoker
1,12.54,2.5,Male,No
2,11.17,1.5,Female,No
3,20.29,2.75,Female,No
"""

# Load your pandas DataFrame
df = pandas.read_csv(StringIO(data_left))

# Obtain client object
client = ClientEngine(keys_path="my_keys")

# Encrypt the DataFrame
df_encrypted = client.encrypt_from_pandas(df)

# Decrypt the DataFrame to produce a pandas DataFrame
df_decrypted = client.decrypt_to_pandas(df_encrypted)
```

## Supported Data Types and Schema Definition

Concrete ML's encrypted `DataFrame` operations support a specific set of data types:

- **Integer**:  Integers are supported within a specific range determined by the encryption scheme's quantization parameters. Default range is 1 to 15. 0 being used for the `NaN`. Values outside this range will cause a `ValueError` to be raised during the pre-processing stage.
- **Quantized Float**: Floating-point numbers are quantized to integers within the supported range. This is achieved by computing a scale and zero point for each column, which are used to map the floating-point numbers to the quantized integer space.
- **String Enum**: String columns are mapped to integers starting from 1. This mapping is stored and later used for de-quantization. If the number of unique strings exceeds 15, a `ValueError` is raised.

## Supported Operations on Encrypted Data-frames

> **Outsourced execution**: The merge operation on Encrypted DataFrames can be **securely** performed on a third-party server. This means that the server can execute the merge without ever having access to the unencrypted data. The server only requires the encrypted DataFrames.

Encrypted DataFrames support a subset of operations that are available for pandas DataFrames. The following operations are currently supported:

- `merge`: left or right join two data-frames

<!--pytest-codeblocks:cont-->

```python
df_right = """index,day,time,size
2,Thur,Lunch,2
5,Sat,Dinner,3
9,Sun,Dinner,2"""

# Encrypt the DataFrame
df_encrypted2 = client.encrypt_from_pandas(pandas.read_csv(StringIO(df_right)))

df_encrypted_merged = df_encrypted.merge(df_encrypted2, how="left", on="index")
```

## Serialization of Encrypted Data-frames

Encrypted `DataFrame` objects can be serialized to a file format for storage or transfer. When serialized, they contain the encrypted data and [evaluation keys](../getting-started/concepts.md#cryptography-concepts) necessary to perform computations.

> **Security**: Serialized data-frames do not contain any secret keys. The data-frames can be exchanged with any third-party without any risk.

### Saving and loading Data-frames

To save or load an encrypted `DataFrame` from a file, use the following commands:

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.pandas import load_encrypted_dataframe

# Save
df_encrypted_merged.save("df_encrypted_merged")

# Load
df_encrypted_merged = load_encrypted_dataframe("df_encrypted_merged")

# Decrypt the DataFrame
df_decrypted = client.decrypt_to_pandas(df_encrypted)
```

## Error Handling

The library is designed to raise specific errors when encountering issues during the pre-processing and post-processing stages:

- `ValueError`: Raised when a column contains values outside the allowed range for integers, when there are too many unique strings, or when encountering an unsupported data type. Raised also when an operation is attempted on a data type that is not supported by the operation.

## Example Workflow

An example workflow where two clients encrypt two `DataFrame` objects, perform a merge operation on the server side, and then decrypt the results is available in the notebook [encrypted_pandas.ipynb](../advanced_examples/EncryptedPandas.ipynb).

## Current Limitations

While this API offers a new secure way to work on remotely stored and encrypted data, it has some strong limitations at the moment:

- **Precision of Values**: The precision for numerical values is limited to 4 bits.
- **Supported Operations**: The `merge` operation is the only one available.
- **Index Handling**: Index values are not preserved; users should move any relevant data from the index to a dedicated new column before encrypting.
- **Integer Range**: The range of integers that can be encrypted is between 1 and 15.
- **Uniqueness for `merge`**: The `merge` operation requires that the columns to merge on contain unique values. Currently this means that data-frames are limited to 15 rows.
- **Metadata Security**: Column names and the mapping of strings to integers are not encrypted and are sent to the server in clear text.
