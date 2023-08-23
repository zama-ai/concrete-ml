<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.deployment.deploy_to_aws`

Methods to deploy a client/server to AWS.

It takes as input a folder with:
\- client.zip
\- server.zip
\- processing.json

It spawns a AWS EC2 instance with proper security groups. Then SSHs to it to rsync the files and update Python dependencies. It then launches the server.

## **Global Variables**

- **DATE_FORMAT**
- **DEFAULT_CML_AMI_ID**

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_instance`

```python
create_instance(
    instance_type: str = 'c5.large',
    open_port=5000,
    instance_name: Optional[str] = None,
    verbose: bool = False,
    region_name: Optional[str] = None,
    ami_id='ami-0d7427e883fa00ff3'
) → Dict[str, Any]
```

Create a EC2 instance.

**Arguments:**

- <b>`instance_type`</b> (str):  the type of AWS EC2 instance.
- <b>`open_port`</b> (int):  the port to open.
- <b>`instance_name`</b> (Optional\[str\]):  the name to use for AWS created objects
- <b>`verbose`</b> (bool):  show logs or not
- <b>`region_name`</b> (Optional\[str\]):  AWS region
- <b>`ami_id`</b> (str):  AMI to use

**Returns:**

- <b>`Dict[str, Any]`</b>:  some information about the newly created instance.
  \- ip
  \- private_key
  \- instance_id
  \- key_path
  \- ip_address
  \- port

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `deploy_to_aws`

```python
deploy_to_aws(
    instance_metadata: Dict[str, Any],
    path_to_model: Path,
    number_of_ssh_retries: int = -1,
    wait_bar: bool = False,
    verbose: bool = False
)
```

Deploy a model to a EC2 AWS instance.

**Arguments:**

- <b>`instance_metadata`</b> (Dict\[str, Any\]):  the metadata of AWS EC2 instance  created using AWSInstance or create_instance
- <b>`path_to_model`</b> (Path):  the path to the serialized model
- <b>`number_of_ssh_retries`</b> (int):  the number of ssh retries (-1 is no limit)
- <b>`wait_bar`</b> (bool):  whether to show a wait bar when waiting for ssh connection to be available
- <b>`verbose`</b> (bool):  whether to show a logs

**Returns:**
instance_metadata (Dict\[str, Any\])

**Raises:**

- <b>`RuntimeError`</b>:  if launching the server crashed

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L422"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wait_instance_termination`

```python
wait_instance_termination(instance_id: str, region_name: Optional[str] = None)
```

Wait for AWS EC2 instance termination.

**Arguments:**

- <b>`instance_id`</b> (str):  the id of the AWS EC2 instance to terminate.
- <b>`region_name`</b> (Optional\[str\]):  AWS region (Optional)

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `terminate_instance`

```python
terminate_instance(instance_id: str, region_name: Optional[str] = None)
```

Terminate a AWS EC2 instance.

**Arguments:**

- <b>`instance_id`</b> (str):  the id of the AWS EC2 instance to terminate.
- <b>`region_name`</b> (Optional\[str\]):  AWS region (Optional)

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `delete_security_group`

```python
delete_security_group(security_group_id: str, region_name: Optional[str] = None)
```

Terminate a AWS EC2 instance.

**Arguments:**

- <b>`security_group_id`</b> (str):  the id of the AWS EC2 instance to terminate.
- <b>`region_name`</b> (Optional\[str\]):  AWS region (Optional)

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `main`

```python
main(
    path_to_model: Path,
    port: int = 5000,
    instance_type: str = 'c5.large',
    instance_name: Optional[str] = None,
    verbose: bool = False,
    wait_bar: bool = False,
    terminate_on_shutdown: bool = True
)
```

Deploy a model.

**Arguments:**

- <b>`path_to_model`</b> (Path):  path to serialized model to serve.
- <b>`port`</b> (int):  port to use.
- <b>`instance_type`</b> (str):  type of AWS EC2 instance to use.
- <b>`instance_name`</b> (Optional\[str\]):  the name to use for AWS created objects
- <b>`verbose`</b> (bool):  show logs or not
- <b>`wait_bar`</b> (bool):  show progress bar when waiting for ssh connection
- <b>`terminate_on_shutdown`</b> (bool):  terminate instance when script is over

______________________________________________________________________

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AWSInstance`

AWSInstance.

Context manager for AWS instance that supports ssh and http over one port.

<a href="../../../src/concrete/ml/deployment/deploy_to_aws.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    instance_type: str = 'c5.large',
    open_port=5000,
    instance_name: Optional[str] = None,
    verbose: bool = False,
    terminate_on_shutdown: bool = True,
    region_name: Optional[str] = None,
    ami_id: str = 'ami-0d7427e883fa00ff3'
)
```
