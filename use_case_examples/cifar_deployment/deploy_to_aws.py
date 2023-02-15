"""Script to deploy a client/server to AWS.

It takes as input a folder with:
    - client.zip
    - server.zip
    - processing.json
It spawns a AWS EC2 instance with proper security groups.
Then SSHs to it to rsync the files and update python dependencies.
It then launches the server.
"""

import io
import os
import select
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import boto3
import paramiko
from tqdm import tqdm

INTERVAL_BETWEEN_SSH_RETRIES = int(os.environ.get("INTERVAL_BETWEEN_SSH_RETRIES", 2))
NUMBER_OF_SSH_RETRIES = int(os.environ.get("NUMBER_OF_SSH_RETRIES", -1))
DATE_FORMAT: str = "%Y_%m_%d_%H_%M_%S"
SSH_TIMEOUT: int = int(os.environ.get("SSH_TIMEOUT", 10))
INSTANCE_TYPE: str = os.environ.get(
    "INSTANCE_TYPE", "c5.metal"
)  # A big instance is needed to run the network
PORT: int = int(os.environ.get("PORT", "5000"))


def wait_with_progress_bar(seconds: int = 10) -> None:
    for _ in tqdm(range(seconds)):
        time.sleep(1)


def ssh_connect_with_retry(
    client: paramiko.SSHClient,
    ip_address: str,
    private_key: str,
    retries: int = NUMBER_OF_SSH_RETRIES,
):
    privkey = paramiko.RSAKey.from_private_key(io.StringIO(private_key))
    if retries == -1:  # Infinite retry
        while True:
            try:
                print(f"SSH into the instance: {ip_address}")
                client.connect(
                    hostname=ip_address, username="ubuntu", pkey=privkey, timeout=SSH_TIMEOUT
                )
                return
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                print(e)
                wait_with_progress_bar(INTERVAL_BETWEEN_SSH_RETRIES)
    else:
        for _ in range(retries + 1):
            try:
                print(f"SSH into the instance: {ip_address}")
                client.connect(
                    hostname=ip_address, username="ubuntu", pkey=privkey, timeout=SSH_TIMEOUT
                )
                return
            except Exception as e:
                print(e)
                wait_with_progress_bar(INTERVAL_BETWEEN_SSH_RETRIES)
    raise ValueError("Wasn't able to connect to server")


if __name__ == "__main__":
    client = boto3.client("ec2")
    resources = boto3.resource("ec2")

    # Get some information
    hostname = socket.gethostname()
    str_now = datetime.now().strftime(DATE_FORMAT)
    name = f"deploy-cml-{str_now}"

    # Get security group
    response: Dict = client.describe_vpcs()
    vpc_id: str = response.get("Vpcs", [{}])[0].get("VpcId", "")

    # OPTION 1: get fist vpc available
    if vpc_id:
        vpc = vpc = resources.Vpc(vpc_id)
    # OPTION 2: create VPC (might crash if too many VPCs)
    else:
        vpc = resources.create_vpc(CidrBlock="0.0.0.0/0")
        vpc.wait_until_available()

    # Get subnet
    subnets = [elt for elt in vpc.subnets.all()]
    # OPTION 1: create subnet
    if not subnets:  # If no subnet create one
        subnet = vpc.create_subnet(CidrBlock=vpc.cidr_block)
    # OPTION 2: get first subnet
    else:
        subnet = subnets[0]

    # Create security group
    response = client.create_security_group(
        GroupName=f"deploy-cml-{str_now}", Description=f"Deploy CML example {str_now}", VpcId=vpc_id
    )
    security_group_id = response["GroupId"]
    print(f"Security Group Created {security_group_id} in vpc {vpc_id}.")
    data = client.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            # Server port
            {
                "IpProtocol": "tcp",
                "FromPort": PORT,
                "ToPort": PORT,
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
    keypair_name = f"deploy-cml-keypair-{str_now}"
    keypair_name = keypair_name
    response = client.create_key_pair(KeyName=keypair_name)
    private_key: str = response["KeyMaterial"]
    private_key = private_key

    # Keep the key if we want to ssh to check what happenned on the instance
    key_folder = (Path(__file__).parent / "ssh_keys").resolve()
    key_folder.mkdir(exist_ok=True)
    key_path = key_folder / f"{keypair_name}.pem"
    with open(key_path, "w", encoding="utf-8") as file:
        file.write(private_key)
    key_path.chmod(0o400)

    # Create instance
    num_instances = 1
    instances = resources.create_instances(
        # CML official AMI ID
        ImageId="ami-0d7427e883fa00ff3",
        # Update with the wanted type of instance
        InstanceType=INSTANCE_TYPE,  # Instance type
        # DryRun=True is used to checked permissions
        DryRun=False,
        # Allows us to just shutdown the instance to terminate it
        # seems like it doesn't always work
        InstanceInitiatedShutdownBehavior="terminate",
        # Associate keypair to instance
        KeyName=keypair_name,
        # Some tags
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Team", "Value": "cml"},
                    {"Key": "Repository", "Value": "concrete-ml-internal"},
                    {"Key": "Name", "Value": name},
                    {"Key": "Project", "Value": "deployment_demo_v0"},
                ],
            },
        ],
        MaxCount=num_instances,
        MinCount=num_instances,
        NetworkInterfaces=[
            {
                "AssociatePublicIpAddress": True,
                "DeviceIndex": 0,
                "SubnetId": subnet.id,
                "Groups": [security_group_id],
            }
        ],
    )

    # Now we can get our instance
    if len(instances) != 1:  # Some sanity check
        raise ValueError(
            "Incorrect number of instances: {len(instances)} ({[elt.id for elt in instances]})"
        )
    instance = instances[0]
    instance_id: str = instance.id

    before_time = time.time()
    instance.wait_until_running()
    instance.reload()  # Needed to update information like public_ip_address
    after_time = time.time()
    print(f"instance took {after_time - before_time} seconds to start running")

    # Get information about instance
    instances = client.describe_instances()
    instance_info = [
        elt
        for elt in instances["Reservations"]
        for sub_elt in elt["Instances"]
        if sub_elt["InstanceId"] == instance.instance_id
    ][0]
    ip_address: str = instance.public_ip_address
    # ip_address: str = instance_info["Instances"][0]["NetworkInterfaces"][0]["Association"]["PublicIp"]

    print(f"ip_address={ip_address}")
    assert isinstance(ip_address, str), "ip adress is None"

    with open("connect_to_instance.sh", "w", encoding="utf-8") as file:
        file.write(
            f"""#! /bin/env bash
ssh -i {key_path.resolve()} ubuntu@{ip_address}"""
        )

    # Connect to instance
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    assert isinstance(instance.public_ip_address, str)
    assert isinstance(private_key, str)

    ssh_connect_with_retry(
        client=ssh_client,
        ip_address=instance.public_ip_address,
        retries=NUMBER_OF_SSH_RETRIES,
        private_key=private_key,
    )

    # Rsync needed files
    subprocess.check_output(
        f"rsync -Pav -e 'ssh -i {key_path.resolve()} -o \"StrictHostKeyChecking=no\"' "
        f"./dev ./server.py ./server_requirements.txt   ubuntu@{ip_address}:~",
        shell=True,
    )

    # Launch command
    commands = [
        "sudo chmod -R 777 /home/ubuntu/venv",  # Needed because of the way the AMI is setup
        # Somehow the venv is not activated by default
        "source venv/bin/activate",
        # Check that all files are here
        "ls",
        "python -m pip install -r server_requirements.txt",
        "python -m pip install concrete-compiler==0.23.2",  # Downgrade for compatibility with docker image
        # Some library version checks
        "python -m pip show concrete-ml",
        "python -m pip show concrete-compiler",
        "python -m pip show concrete-numpy",
        "python ./server.py",
    ]

    final_command = " && ".join(commands)
    print(f"Command: {final_command}")
    stdin, stdout, stderr = ssh_client.exec_command(command=final_command)  # This does not block
    instance.terminate_on_exception = False

    print("Passed this point you will be shown AWS logs")
    print("You can now CTRL+C at any moment without disturbing the execution of your script.")

    print("LOGS:")
    # Process output of instance
    stdout.channel.set_combine_stderr(True)
    stdin.close()  # We don't stdin
    channel = stdout.channel
    channel.set_combine_stderr(True)
    # indicate that we're not going to write to that channel anymore
    channel.shutdown_write()
    # read stdout/stderr in order to prevent read block hangs
    timeout = 10
    stdout_chunks: List[bytes] = []
    stderr_chunks: List[bytes] = []
    stdout_chunks.append(stdout.channel.recv(len(stdout.channel.in_buffer)))
    print(stdout_chunks[-1].decode(encoding="utf-8", errors="ignore"))

    # chunked read to prevent stalls
    while not channel.closed or channel.recv_ready() or channel.recv_stderr_ready():
        """
        1) make sure that there are at least 2 cycles with no data in the input buffers in order to not exit too early (i.e. cat on a >200k file).
        2) if no data arrived in the last loop, check if we already received the exit code
        3) check if input buffers are empty
        4) exit the loop
        """
        try:
            # stop if channel was closed prematurely, and there is no data in the buffers.
            got_chunk = False
            readq, _, _ = select.select([stdout.channel], [], [], timeout)
            for c in readq:
                if c.recv_ready():
                    stdout_chunks.append(stdout.channel.recv(len(c.in_buffer)))
                    print(stdout_chunks[-1].decode(encoding="utf-8", errors="ignore"))
                    got_chunk = True

                if c.recv_stderr_ready():
                    # make sure to read stderr to prevent stall
                    # + tqdm logs to stderr
                    stderr_chunks.append(stderr.channel.recv_stderr(len(c.in_stderr_buffer)))
                    print(stderr_chunks[-1].decode(encoding="utf-8", errors="ignore"))
                    got_chunk = True
            if (
                not got_chunk
                and stdout.channel.exit_status_ready()
                and not stderr.channel.recv_stderr_ready()
                and not stdout.channel.recv_ready()
            ):
                print("breaking")
                # indicate that we're not going to read from this channel anymore
                stdout.channel.shutdown_read()
                # close the channel
                stdout.channel.close()
                raise KeyboardInterrupt()

        # Handle keyboard interrupt
        except KeyboardInterrupt as exception:
            message = "Terminate instance? [Yes,No,Continue]"
            terminate_instance = input(message)
            terminate_instance = terminate_instance.lower()
            while terminate_instance not in {"y", "n", "yes", "no", "c", "continue"}:
                terminate_instance = input(message)
                terminate_instance = terminate_instance.lower()
            if terminate_instance in {"y", "yes"}:
                client.terminate_instances(InstanceIds=[instance_id])
                stdout.close()
                stderr.close()
                sys.exit(0)
            if terminate_instance in {"n", "no"}:
                # close all the pseudofiles
                stdout.close()
                stderr.close()
                sys.exit(0)
            if terminate_instance in {"c", "continue"}:
                print(f"URL=http://{ip_address}:{PORT}")
                continue
