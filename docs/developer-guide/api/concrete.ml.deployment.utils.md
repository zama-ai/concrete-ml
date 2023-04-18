<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/deployment/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.utils`

Utils.

- Check if connection possible
- Wait for connection to be available (with timeout)

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/utils.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `filter_logs`

```python
filter_logs(previous_logs: str, current_logs: str) → str
```

Filter logs based on previous logs.

**Arguments:**

- <b>`previous_logs`</b> (str):  previous logs
- <b>`current_logs`</b> (str):  current logs

**Returns:**

- <b>`str`</b>:  filtered logs

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/utils.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wait_for_connection_to_be_available`

```python
wait_for_connection_to_be_available(
    hostname: str,
    ip_address: str,
    path_to_private_key: Path,
    timeout: int = 1,
    wait_time: int = 1,
    max_retries: int = 20,
    wait_bar: bool = False
)
```

Wait for connection to be available.

**Arguments:**

- <b>`hostname`</b> (str):  host name
- <b>`ip_address`</b> (str):  ip address
- <b>`path_to_private_key`</b> (Path):  path to private key
- <b>`timeout`</b> (int):  ssh timeout option
- <b>`wait_time`</b> (int):  time to wait between retries
- <b>`max_retries`</b> (int):  number of retries, if \< 0 unlimited retries
- <b>`wait_bar`</b> (bool):  tqdm progress bar of retries

**Raises:**

- <b>`TimeoutError`</b>:  if it wasn't able connect to ssh with the given constraints

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/utils.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_connection_available`

```python
is_connection_available(
    hostname: str,
    ip_address: str,
    path_to_private_key: Path,
    timeout: int = 1
)
```

Check if ssh connection is available.

**Arguments:**

- <b>`hostname`</b> (str):  host name
- <b>`ip_address`</b> (str):  ip address
- <b>`path_to_private_key`</b> (Path):  path to private key
- <b>`timeout`</b>:  ssh timeout option

**Returns:**

- <b>`bool`</b>:  True if connection succeeded
