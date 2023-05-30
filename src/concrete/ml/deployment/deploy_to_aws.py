"""Methods to deploy a client/server to AWS.

It takes as input a folder with:
    - client.zip
    - server.zip
    - processing.json

It spawns a AWS EC2 instance with proper security groups.
Then SSHs to it to rsync the files and update Python dependencies.
It then launches the server.
"""

import argparse
import json
import subprocess
import time
import uuid
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3

from ..deployment.utils import filter_logs, wait_for_connection_to_be_available

DATE_FORMAT: str = "%Y_%m_%d_%H_%M_%S"
# More up to date public Concrete ML AWS AMI
DEFAULT_CML_AMI_ID: str = "ami-0d7427e883fa00ff3"


class AWSInstance:
    """AWSInstance.

    Context manager for AWS instance that supports ssh and http over one port.
    """

    instance_metadata: Dict[str, Any]

    def __init__(
        self,
        instance_type: str = "c5.large",
        open_port=5000,
        instance_name: Optional[str] = None,
        verbose: bool = False,
        terminate_on_shutdown: bool = True,
        region_name: Optional[str] = None,
        ami_id: str = DEFAULT_CML_AMI_ID,
    ):
        metadata = create_instance(
            instance_type=instance_type,
            open_port=open_port,
            instance_name=instance_name,
            verbose=verbose,
            region_name=region_name,
            ami_id=ami_id,
        )
        self.instance_metadata = metadata
        self.terminate_on_shutdown = terminate_on_shutdown
        self.region_name = region_name

    def __enter__(
        self,
    ):
        """Return instance_metadata and streams.

        Returns:
            Dict[str, Any]
                - ip
                - private_key
                - instance_id
                - key_path
                - ip_address
        """
        return self.instance_metadata

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Terminates the instance.

        Arguments:
            exc_type: exception type
            exc_value: exception value
            exc_traceback: exception traceback
        """
        if self.terminate_on_shutdown:
            terminate_instance(self.instance_metadata["instance_id"], region_name=self.region_name)
            # We need to wait for instance termination to delete the security group
            wait_instance_termination(
                self.instance_metadata["instance_id"], region_name=self.region_name
            )
            delete_security_group(
                self.instance_metadata["security_group_id"], region_name=self.region_name
            )


def create_instance(
    instance_type: str = "c5.large",
    open_port=5000,
    instance_name: Optional[str] = None,
    verbose: bool = False,
    region_name: Optional[str] = None,
    ami_id=DEFAULT_CML_AMI_ID,
) -> Dict[str, Any]:
    """Create a EC2 instance.

    Arguments:
        instance_type (str): the type of AWS EC2 instance.
        open_port (int): the port to open.
        instance_name (Optional[str]): the name to use for AWS created objects
        verbose (bool): show logs or not
        region_name (Optional[str]): AWS region
        ami_id (str): AMI to use

    Returns:
        Dict[str, Any]: some information about the newly created instance.
            - ip
            - private_key
            - instance_id
            - key_path
            - ip_address
            - port
    """
    open_port = int(open_port)

    # Create client/resource objects
    with closing(boto3.client("ec2", region_name=region_name)) as client:
        resources = boto3.resource("ec2", region_name=region_name)
        str_now = datetime.now().strftime(DATE_FORMAT)
        name = (
            f"deploy-cml-{str_now}-{uuid.uuid4()}"
            if instance_name is None
            else f"{instance_name}-{str_now}"
        )

        # Get VPC
        vpc_id: str = client.describe_vpcs().get("Vpcs", [{}])[0].get("VpcId", "")
        # OPTION 1: get fist vpc available
        if vpc_id:
            vpc = resources.Vpc(vpc_id)
        # OPTION 2: create VPC (not possible if too many VPCs)
        else:  # pragma:no cover
            vpc = resources.create_vpc(CidrBlock="0.0.0.0/0")
            vpc.wait_until_available()

        # Get subnet
        subnets = list(vpc.subnets.all())
        # OPTION 1: create subnet
        if not subnets:  # pragma: no cover
            subnet = vpc.create_subnet(CidrBlock=vpc.cidr_block)
        # OPTION 2: get first subnet
        else:
            subnet = subnets[0]

        # Create security group
        security_group_id = client.create_security_group(
            GroupName=name, Description=f"Deploy Concrete ML {str_now}", VpcId=vpc_id
        )["GroupId"]
        if verbose:
            print(f"Security Group Created {security_group_id} in vpc {vpc_id}.")

        client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                # Server port
                {
                    "IpProtocol": "tcp",
                    "FromPort": open_port,
                    "ToPort": open_port,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                },
                # SSH port
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                },
            ],
        )

        # Create key-pair
        keypair_name = f"{name}-keypair"
        private_key: str = client.create_key_pair(KeyName=keypair_name)["KeyMaterial"]

        # Keep the key if we want to ssh to check what happened on the instance
        key_folder = (Path(__file__).parent / "ssh_keys").resolve()
        key_folder.mkdir(exist_ok=True)
        key_path = key_folder / f"{keypair_name}.pem"
        with open(key_path, "w", encoding="utf-8") as file:
            file.write(private_key)
        key_path.chmod(0o400)

        # Create instance
        instances = resources.create_instances(
            # Concrete ML official AMI ID to make sure to have everything needed
            ImageId=ami_id,
            InstanceType=instance_type,  # Instance type
            DryRun=False,
            InstanceInitiatedShutdownBehavior="terminate",
            # Associate keypair to instance
            KeyName=keypair_name,
            # Some tags
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Repository", "Value": "concrete-ml"},
                        {"Key": "Name", "Value": name},
                    ],
                },
            ],
            # Number of instances
            MaxCount=1,
            MinCount=1,
            NetworkInterfaces=[
                {
                    "AssociatePublicIpAddress": True,
                    "DeviceIndex": 0,
                    "SubnetId": subnet.id,
                    "Groups": [security_group_id],
                }
            ],
        )

        instance = instances[0]
        instance.terminate_on_exception = False

        before_time = time.time()
        instance.wait_until_running()
        instance.reload()  # Needed to update information like public_ip_address
        if verbose:
            print(f"Instance took {time.time() - before_time} seconds to start running")

        # Get information about instance
        ip_address: str = instance.public_ip_address
        assert ip_address is not None

        metadata: Dict[str, Any] = {
            "ip": ip_address,
            "private_key": private_key,
            "instance_id": instance.id,
            "key_path": key_path,
            "ip_address": ip_address,
            "port": open_port,
            "instance_name": name,
            "security_group_id": security_group_id,
        }

    return metadata


def deploy_to_aws(
    instance_metadata: Dict[str, Any],
    path_to_model: Path,
    number_of_ssh_retries: int = -1,
    wait_bar: bool = False,
    verbose: bool = False,
):
    """Deploy a model to a EC2 AWS instance.

    Arguments:
        instance_metadata (Dict[str, Any]): the metadata of AWS EC2 instance
            created using AWSInstance or create_instance
        path_to_model (Path): the path to the serialized model
        number_of_ssh_retries (int): the number of ssh retries (-1 is no limit)
        wait_bar (bool): whether to show a wait bar when waiting for ssh connection to be available
        verbose (bool): whether to show a logs

    Returns:
        instance_metadata (Dict[str, Any])

    Raises:
        RuntimeError: if launching the server crashed
    """

    port = instance_metadata["port"]
    ip_address: str = instance_metadata["ip_address"]
    key_path: Path = instance_metadata["key_path"]
    instance_metadata["URL"] = f"http://{ip_address}:{port}"
    hostname = "ubuntu"

    with open("connect_to_instance.sh", "w", encoding="utf-8") as file:
        file.write(
            f"""#! /bin/env bash
ssh -i {key_path.resolve()} {hostname}@{ip_address}"""
        )

    with open("terminate_instance.sh", "w", encoding="utf-8") as file:
        file.write(
            f"""#! /bin/env bash
aws ec2 terminate-instances --instance-ids {instance_metadata['instance_id']}
aws ec2 delete-security-group --group-id {instance_metadata['security_group_id']}"""
        )

    if verbose:
        print("Waiting for SSH connection to be available...")

    # Connect to instance
    wait_for_connection_to_be_available(
        hostname=hostname,
        ip_address=ip_address,
        path_to_private_key=key_path,
        timeout=1,
        max_retries=number_of_ssh_retries,
        wait_bar=wait_bar,
    )

    if verbose:
        print("SSH connection available.")

    path_to_server_file = Path(__file__).parent / "server.py"
    path_to_server_requirements = Path(__file__).parent / "server_requirements.txt"

    if verbose:
        print("upload files...")

    # Rsync needed files
    subprocess.check_output(
        f"rsync -Pav -e 'ssh -i {key_path.resolve()} "
        '-o "StrictHostKeyChecking=no" -o " IdentitiesOnly=yes"\' '
        f"{path_to_model.resolve()} {path_to_server_file.resolve()} "
        f"{path_to_server_requirements.resolve()}   {hostname}@{ip_address}:~",
        shell=True,
    )

    if verbose:
        print("upload finished.")

    # Load versions for checking
    with zipfile.ZipFile(path_to_model.resolve().joinpath("client.zip")) as client_zip:
        with client_zip.open("versions.json", mode="r") as file:
            versions = json.load(file)

    python_version = ".".join(versions["python"].split(".")[0:2])
    concrete_python_version = versions["concrete-python"]
    concrete_ml_version = versions["concrete-ml"]

    # Launch commands
    commands = [
        # Needed because of the way the AMI is setup
        f"sudo chmod -R 777 /home/{hostname}/venv",
        f"sudo apt install -y python{python_version} python{python_version}-distutils make cmake",
        f"virtualenv deployment_venv --python=python{python_version}",
        # The venv is not activated by default
        "source deployment_venv/bin/activate",
        # Install server requirements
        "python -m pip install -r server_requirements.txt",
        # We need to relax the constraint on the version for internal testing
        f"python -m pip install concrete-ml=={concrete_ml_version}"
        " || python -m pip install concrete-ml",
        # We still need to force concrete-python version to be exactly the same as the file
        f"python -m pip install concrete-python=={concrete_python_version} || :",
        # Launch server
        f'PORT={port} PATH_TO_MODEL="./{path_to_model.name}" python ./server.py',
    ]

    # + f"-o RemoteCommand=\"tmux new -A -s {instance_metadata['instance_name']}\" "
    # Join commands
    ssh_command = (
        f"ssh -i {key_path.resolve()} "
        + "-o StrictHostKeyChecking=no "
        + "-o IdentitiesOnly=yes "
        + "-o RequestTTY=yes "
        + f"{hostname}@{ip_address} "
    )
    launch_command = (
        ssh_command
        + f'"tmux new-session -d -s {instance_metadata["instance_name"]} '
        + "'"
        + " && ".join(commands)
        + " || sleep 10"
        + "'"
        + '"'
    )
    monitoring_command = (
        ssh_command + f"tmux capture-pane -pt {instance_metadata['instance_name']}:0.0"
    )

    # Launch
    subprocess.check_output(launch_command, shell=True, stderr=subprocess.STDOUT)

    last_tmux_logs = ""

    while True:
        try:
            tmux_logs = subprocess.check_output(
                monitoring_command,
                shell=True,
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exception:  # pragma: no cover
            raise RuntimeError(
                "Something crashed when launching the server.\n" f" Last logs:\n{last_tmux_logs}"
            ) from exception

        if any(
            error_message in tmux_logs
            for error_message in ["can't find session:", "no server running on"]
        ):  # pragma: no cover
            raise RuntimeError(
                "Something crashed when launching the server.\n" f" Last logs:\n{last_tmux_logs}"
            )

        if verbose:
            # This could be improved
            to_print = filter_logs(current_logs=tmux_logs, previous_logs=last_tmux_logs).strip()
            if to_print:
                print(to_print)

        # Monitor and return correct
        if "0.0.0.0" in tmux_logs:
            break

        last_tmux_logs = tmux_logs
        time.sleep(1)  # Wait one second
    return instance_metadata


def wait_instance_termination(instance_id: str, region_name: Optional[str] = None):
    """Wait for AWS EC2 instance termination.

    Arguments:
        instance_id (str): the id of the AWS EC2 instance to terminate.
        region_name (Optional[str]): AWS region (Optional)
    """
    with closing(boto3.client("ec2", region_name=region_name)) as client:
        waiter = client.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=[instance_id])


def terminate_instance(instance_id: str, region_name: Optional[str] = None):
    """Terminate a AWS EC2 instance.

    Arguments:
        instance_id (str): the id of the AWS EC2 instance to terminate.
        region_name (Optional[str]): AWS region (Optional)
    """
    with closing(boto3.client("ec2", region_name=region_name)) as client:
        client.terminate_instances(InstanceIds=[instance_id])


def delete_security_group(security_group_id: str, region_name: Optional[str] = None):
    """Terminate a AWS EC2 instance.

    Arguments:
        security_group_id (str): the id of the AWS EC2 instance to terminate.
        region_name (Optional[str]): AWS region (Optional)
    """
    with closing(boto3.client("ec2", region_name=region_name)) as client:
        client.delete_security_group(GroupId=security_group_id)


def main(
    path_to_model: Path,
    port: int = 5000,
    instance_type: str = "c5.large",
    instance_name: Optional[str] = None,
    verbose: bool = False,
    wait_bar: bool = False,
    terminate_on_shutdown: bool = True,
):  # pragma: no cover
    """Deploy a model.

    Arguments:
        path_to_model (Path): path to serialized model to serve.
        port (int): port to use.
        instance_type (str): type of AWS EC2 instance to use.
        instance_name (Optional[str]): the name to use for AWS created objects
        verbose (bool): show logs or not
        wait_bar (bool): show progress bar when waiting for ssh connection
        terminate_on_shutdown (bool): terminate instance when script is over
    """

    with AWSInstance(
        instance_type=instance_type,
        open_port=port,
        instance_name=instance_name,
        verbose=verbose,
        terminate_on_shutdown=terminate_on_shutdown,
    ) as instance_metadata:
        instance_metadata = deploy_to_aws(
            instance_metadata=instance_metadata,
            number_of_ssh_retries=-1,
            path_to_model=path_to_model,
            verbose=verbose,
            wait_bar=wait_bar,
        )
        url = f"http://{instance_metadata['ip_address']}:{port}"
        print(url + "\r\n", end="", sep="")
        with open("url.txt", mode="w", encoding="utf-8") as file:
            file.write(url)

        print(f"Server running at {url} .\nNow waiting indefinitely.\r\n", end="", sep="")

        while True:
            time.sleep(1)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-model", dest="path_to_model", type=Path, default=Path("./dev"))
    parser.add_argument("--port", dest="port", type=str, default="5000")
    parser.add_argument("--instance-type", dest="instance_type", type=str, default="c5.large")
    parser.add_argument("--instance-name", dest="instance_name", type=str, default="cml-deploy")
    parser.add_argument("--verbose", dest="verbose", type=lambda elt: bool(int(elt)), default=False)
    parser.add_argument(
        "--terminate-on-shutdown",
        dest="terminate_on_shutdown",
        type=lambda elt: bool(int(elt)),
        default=True,
    )
    parser.add_argument(
        "--wait-bar", dest="wait_bar", type=lambda elt: bool(int(elt)), default=False
    )
    args = parser.parse_args()

    main(
        path_to_model=args.path_to_model,
        port=args.port,
        instance_type=args.instance_type,
        instance_name=args.instance_name,
        verbose=args.verbose,
        wait_bar=args.wait_bar,
        terminate_on_shutdown=args.terminate_on_shutdown,
    )
