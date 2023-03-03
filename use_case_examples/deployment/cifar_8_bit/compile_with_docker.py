import os
import shutil
import subprocess
from pathlib import Path


def main():
    path_of_script = Path(__file__).parent.resolve()
    files = [
        "encrypted_module.py",
        "clear_module.py",
        "model.py",
        "brevitas_utils.py",
        "8_bit_model.pt",
        "constants.py",
    ]

    # Copy files
    for file_name in files:
        source = Path(
            path_of_script / f"../../cifar_brevitas_with_model_splitting/{file_name}"
        ).resolve()
        target = Path(path_of_script / file_name).resolve()
        if not target.exists():
            print(f"{source} -> {target}")
            shutil.copyfile(src=source, dst=target)

    # Build image
    os.chdir(path_of_script)
    command = f'docker build --tag compile_cifar --file "{path_of_script}/Dockerfile.compile" . && \
docker run --name compile_cifar compile_cifar && \
docker cp compile_cifar:/project/dev . && \
docker rm "$(docker ps -a --filter name=compile_cifar -q)"'
    subprocess.check_output(command, shell=True)

    # Remove files
    for file_name in files:
        target = Path(path_of_script / file_name).resolve()
        target.unlink()


if __name__ == "__main__":
    main()
