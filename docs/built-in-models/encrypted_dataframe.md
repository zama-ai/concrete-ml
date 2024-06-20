# Working with Encrypted DataFrames

This document introduces how to construct and perform operations on encrypted DataFrames using Fully Homomorphic Encryption (FHE).

## Introduction

Encrypted DataFrames are a storage format for encrypted tabular data. You can exchange encrypted DataFrames with third parties to collaborate without privacy risks. Potential applications include:

- Encrypt storage of tabular data-sets
- Joint data analysis efforts between multiple parties
- Data preparation steps before machine learning tasks, such as inference or training
- Secure outsourcing of data analysis to untrusted third parties

## Encryption and decryption

To encrypt a pandas DataFrame, construct a `ClientEngine` that manages keys and then call the `encrypt_from_pandas` function:

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

## Supported data types and schema definition

- **Integer**:  Integers are supported within a specific range determined by the encryption scheme's quantization parameters. Default range is 1 to 15. 0 being used for the `NaN`. Values outside this range will cause a `ValueError` to be raised during the pre-processing stage.
- **Quantized Float**: Floating-point numbers are quantized to integers within the supported range. This is achieved by computing a scale and zero point for each column, which are used to map the floating-point numbers to the quantized integer space.
- **String Enum**: String columns are mapped to integers starting from 1. This mapping is stored and later used for de-quantization. If the number of unique strings exceeds 15, a `ValueError` is raised.

### Using a user-defined schema

Before encryption, the data is preprocessed. For example **string enums** first need to be mapped to integers, and floating point values must be quantized. By default, this mapping is done automatically. However, when two different clients encrypt their data separately, the automatic mappings may differ, possibly due to some missing values in one of the client's DataFrame. Thus the column can not be selected when merging encrypted DataFrames.

The encrypted DataFrame supports user-defined mappings. These schemas are defined as a dictionary where keys represent column names and values contain meta-data about the column. Supported column meta-data are:

- string columns: mapping between string values and integers.
- float columns: the min/max range that the column values lie in.

<!--pytest-codeblocks:cont-->

```python
schema = {
    "string_column": {"abc": 1, "bcd": 2 },
    "float_column": {"min": 0.1, "max": 0.5 }
}
```

## Supported operations

Encrypted DataFrame is designed to support a subset of operations that are available for pandas DataFrames. For now, only the `merge` operation is supported. More operations will be added in the future releases.

### Merge operation

Merge operation allows you to left or right join two DataFrames.

> \[!NOTE\]\
> The merge operation on Encrypted DataFrames can be **securely** performed on a third-party server, meaning that the server can execute the merge without ever having access to the unencrypted data. The server only requires the encrypted DataFrames.

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

## Serialization

You can serialize encrypted DataFrame objects to a file format for storage or transfer. When serialized, they contain the encrypted data and [public evaluation keys](../getting-started/concepts.md#cryptography-concepts) necessary to perform computations.

> \[!NOTE\]\
> Serialized DataFrames do not contain any [private encryption keys](../getting-started/concepts.md#cryptography-concepts) . The DataFrames can be exchanged with any third-party without any risk.

To save or load an encrypted DataFrame from a file, use the following commands:

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

## Error handling

During the pre-processing and post-processing stages, the `ValueError` can happen in the following situations:

- A column contains values outside the allowed range for integers
- Too many unique strings
- Unsupported data type by Concrete ML
- Unsupported data type by the operation attempted

## Example workflow

An example workflow where two clients encrypt two `DataFrame` objects, perform a merge operation on the server side, and then decrypt the results is available in the notebook [encrypted_pandas.ipynb](../advanced_examples/EncryptedPandas.ipynb).

## Current limitations

While this API offers a new secure way to work on remotely stored and encrypted data, it has some strong limitations at the moment:

- **Precision of Values**: The precision for numerical values is limited to 4 bits.
- **Supported Operations**: The `merge` operation is the only one available.
- **Index Handling**: Index values are not preserved; users should move any relevant data from the index to a dedicated new column before encrypting.
- **Integer Range**: The range of integers that can be encrypted is between 1 and 15.
- **Uniqueness for `merge`**: The `merge` operation requires that the columns to merge on contain unique values. Currently this means that data-frames are limited to 15 rows.
- **Metadata Security**: Column names and the mapping of strings to integers are not encrypted and are sent to the server in clear text.
