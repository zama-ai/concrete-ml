import os
import shutil
import subprocess
from pathlib import Path

import torchvision
import torchvision.transforms as transforms


def main():
    path_of_script = Path(__file__).parent.resolve()
    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load data
    try:
        train_set = torchvision.datasets.CIFAR10(
            root=path_of_script / "data",
            train=True,
            download=False,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    except:
        train_set = torchvision.datasets.CIFAR10(
            root=path_of_script / "data",
            train=True,
            download=True,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    del train_set

    files = ["clear_module.py", "constants.py", "brevitas_utils.py", "clear_module.pt"]

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
    command = f'docker build --tag cml_client_cifar_10_8_bit --file "{path_of_script}/Dockerfile.client" .'
    print(command)
    subprocess.check_output(command, shell=True)

    # Remove files
    for file_name in files:
        target = Path(path_of_script / file_name).resolve()
        target.unlink()


if __name__ == "__main__":
    main()
