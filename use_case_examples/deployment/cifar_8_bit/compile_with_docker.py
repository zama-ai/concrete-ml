import os
import shutil
import subprocess
from pathlib import Path


def main():
    path_of_script = Path(__file__).parent.resolve()
    files = [
        "models/__init__.py",
        "models/cnv_2w2a.ini",
        "models/common.py",
        "models/model.py",
        "models/tensor_norm.py",
        "experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar",
    ]

    # Copy files
    for file_name in files:
        source = Path(path_of_script / f"../../cifar/cifar_brevitas_training/{file_name}").resolve()
        target = Path(path_of_script / file_name).resolve()
        if not target.exists():
            print(f"{source} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
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
