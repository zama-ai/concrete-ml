import os
import subprocess
from pathlib import Path


def main():
    path_of_script = Path(__file__).parent.resolve()
    # Build image
    os.chdir(path_of_script)
    command = (
        f'docker build --tag cml_client_breast_cancer --file "{path_of_script}/Dockerfile.client" .'
    )
    print(command)
    subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main()
