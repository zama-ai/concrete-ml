"""Script allowing to launch each benchmark on a different AWS instance."""
import os
import subprocess
import time
from pathlib import Path

import boto3
import paramiko
from paramiko.channel import ChannelFile

HOME = str(Path.home())
LOG_DIR = HOME + "/logs/"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

MAX_EXEC_TIME = 180  # minutes


# Utils functions
def check_if_command_running(out: ChannelFile):
    """Chekc if the command is running"""
    return not out.channel.exit_status_ready()


def check_if_command_success(out: ChannelFile):
    """Check if the command is success"""
    return out.channel.recv_exit_status() == 0


def shutdown_instance(running_instance):
    """Shutdown instance"""
    running_instance.terminate()


def clean_command(command_to_clean: str):
    """Clean command"""
    command_to_clean = command_to_clean.replace(r'"', r"\"")
    return command_to_clean


# Load all environment variables into secrets
secrets = os.environ

AWS_REGION = secrets["AWS_REGION"]
AWS_ACCESS_KEY_ID = secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = secrets["AWS_SECRET_ACCESS_KEY"]
BENCHMARKS_EC2_AMI = secrets["BENCHMARKS_EC2_AMI"]
BENCHMARKS_EC2_INSTANCE_TYPE = secrets["BENCHMARKS_EC2_INSTANCE_TYPE"]
AWS_EC2_SUBNET_ID = secrets["AWS_EC2_SUBNET_ID"]
BENCHMARKS_EC2_SECURITY_GROUP_ID = secrets["BENCHMARKS_EC2_SECURITY_GROUP_ID"]
ML_PROGRESS_TRACKER_TOKEN = secrets["ML_PROGRESS_TRACKER_TOKEN"]
ML_PROGRESS_TRACKER_URL = secrets["ML_PROGRESS_TRACKER_URL"]
PIP_INDEX_URL = secrets["PIP_INDEX_URL"]
PIP_EXTRA_INDEX_URL = secrets["PIP_EXTRA_INDEX_URL"]
BENCHMARK_SAMPLES = secrets["BENCHMARK_SAMPLES"]
GITHUB_COMMIT_SHA = secrets["GITHUB_COMMIT_SHA"]
BENCHMARK_FILE = secrets["BENCHMARK_FILE"]
BENCHMARK_DATASET = secrets["BENCHMARK_DATASET"]
BENCHMARK_MODEL = secrets["BENCHMARK_MODEL"]

# Infer the model type from the BENCHMARK_FILE
if BENCHMARK_FILE.startswith("regression") or BENCHMARK_FILE.startswith("glm"):
    MODEL_TYPE = "regressors"
elif BENCHMARK_FILE.startswith("classification"):
    MODEL_TYPE = "classifiers"
else:
    raise ValueError(f"Unknwon BENCHMARK_FILE: {BENCHMARK_FILE}")

COMMAND_START = f"python3 benchmarks/{BENCHMARK_FILE} " f"--{MODEL_TYPE} {BENCHMARK_MODEL} "
if BENCHMARK_DATASET != "":
    COMMAND_START += f"--datasets {BENCHMARK_DATASET} "

# Get all commands using python benchmarks/BENCHMARK_FILE
commands = (
    subprocess.check_output(
        COMMAND_START + "--list",
        shell=True,
    )
    .decode("utf-8")
    .split("\n")
)

# Remove empty lines
processed_commands = []
for command in commands:
    if command != "":
        processed_commands.append(clean_command(command))
commands = processed_commands

# For now we want to have each command run on a different instance
N_INSTANCES = len(commands)

ec2_ressource = boto3.resource(
    "ec2",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Machine specs
MACHINE_NAME = "AWS (c5.2xlarge)"
MACHINE_VCPU = "8"
MACHINE_OS = "Ubuntu 20.04"
BENCHMARK_SAMPLES = "5"
ARTIFACT_NAME = "c5.2xlarge"

# Create instances
instances = ec2_ressource.create_instances(
    ImageId=BENCHMARKS_EC2_AMI,
    InstanceType=BENCHMARKS_EC2_INSTANCE_TYPE,
    MinCount=N_INSTANCES,
    MaxCount=N_INSTANCES,
    KeyName="ml-benchmark",
    SubnetId=AWS_EC2_SUBNET_ID,
    SecurityGroupIds=[BENCHMARKS_EC2_SECURITY_GROUP_ID],
    TagSpecifications=[
        {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "ml-benchmarks_instance"}]}
    ],
)

# Now that machines run, we have everything in a try catch block to
# ensure that we shutdown all instances if something goes wrong.
try:
    # Private key to file
    pkey_file_path = f"{HOME}/ssh-key"
    pkey = paramiko.RSAKey.from_private_key_file(pkey_file_path)

    # Init commands
    my_command = f"""cd ~/project
git clean -dfx
git remote update
echo "Resetting to {GITHUB_COMMIT_SHA}"
git reset --hard {GITHUB_COMMIT_SHA}
echo "Current rev:"
git rev-parse HEAD
rm -rf .env
touch .env
chmod 0600 .env
echo "export PIP_INDEX_URL='{PIP_INDEX_URL}'" >> .env
echo "export PIP_EXTRA_INDEX_URL='{PIP_EXTRA_INDEX_URL}'" >> .env
echo "export PROGRESS_SAMPLES='{BENCHMARK_SAMPLES}'" >> .env
echo "export PROGRESS_MACHINE_NAME='{MACHINE_NAME}'" >> .env
echo "export PROGRESS_MACHINE_VCPU='{MACHINE_VCPU}'" >> .env
echo "export PROGRESS_MACHINE_OS='{MACHINE_OS}'" >> .env
echo "export ML_PROGRESS_TRACKER_URL='{ML_PROGRESS_TRACKER_URL}'" >> .env
echo "export ML_PROGRESS_TRACKER_TOKEN='{ML_PROGRESS_TRACKER_TOKEN}'" >> .env
make docker_clean_volumes"""

    # Get instance IPs and connect
    machines = {}

    # Wait for instances to be ready
    for instance in instances:
        instance.wait_until_running()

        # Reload instance info
        instance.reload()
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        for _ in range(5):
            try:
                ssh.connect(
                    hostname=instance.public_ip_address,
                    username="ubuntu",
                    pkey=pkey,
                )
                break
            except paramiko.ssh_exception.NoValidConnectionsError:
                time.sleep(5)
        else:
            print(f"Could not connect to instance{instance.public_ip_address}")
            continue

        # Launch init command
        stdin, stdout, stderr = ssh.exec_command(my_command)
        machines[instance.public_ip_address] = {
            "ssh": ssh,
            "stdin": stdin,
            "stdout": stdout,
            "stderr": stderr,
            "instance": instance,
            "status": "init",
        }

    for value in machines.values():
        # Wait for machine to be ready
        while check_if_command_running(value["stdout"]):
            time.sleep(1)
        with open(
            f"{LOG_DIR}/{value['instance'].public_ip_address}.log", "w", encoding="utf-8"
        ) as f:
            f.write("STDOUT_INIT\n")
            f.write(value["stdout"].read().decode("utf-8"))
            f.write("\nSTDERR_INIT\n")
            f.write(value["stderr"].read().decode("utf-8"))
        value["status"] = "ready"

    # Time for benchmarks
    for machine, command in zip(machines.values(), commands):
        # Build command
        # Replace " in command by \"
        command = (
            f"cd ~/project\nmake docker_publish_measurements "
            f'LAUNCH_COMMAND="{COMMAND_START} {command}"'
        )

        # Launch command
        stdin, stdout, stderr = machine["ssh"].exec_command(command)
        machine["stdin"] = stdin
        machine["stdout"] = stdout
        machine["stderr"] = stderr
        machine["status"] = "running"

    timer = time.time()

    REMAINING_INSTANCES = N_INSTANCES
    while True:
        # Break if time is over
        if time.time() - timer > MAX_EXEC_TIME * 60 or REMAINING_INSTANCES == 0:
            break

        # Wait for 5 seconds between each check
        time.sleep(5)

        # Check instances that are idle
        for value in machines.values():
            if value["status"] != "offline":
                # Check if the command is running
                if not check_if_command_running(value["stdout"]):
                    # Save stdout and stderr to log
                    with open(
                        f"{LOG_DIR}/{value['instance'].public_ip_address}.log",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write("STDOUT_COMMAND\n")
                        f.write(value["stdout"].read().decode("utf-8"))
                        f.write("\nSTDERR_COMMAND\n")
                        f.write(value["stderr"].read().decode("utf-8"))

                    # Terminate instance
                    shutdown_instance(value["instance"])

                    # Remove from sshs
                    value["status"] = "offline"
                    REMAINING_INSTANCES -= 1

    # Terminate remaining instances
    for instance in instances:

        # If instance still running, shutdown
        if instance.state["Name"] == "running":
            shutdown_instance(instance)


except Exception as e:
    # If something went wrong, shutdown all instances
    for instance in instances:
        shutdown_instance(instance)
    raise e

# A final check to make sure all instances are terminated
for instance in instances:
    instance.reload()
    if instance.state["Name"] == "running":
        shutdown_instance(instance)

        # Wait for instance to be terminated
        instance.wait_until_terminated()
