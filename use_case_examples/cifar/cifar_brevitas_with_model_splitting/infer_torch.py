"""Run the model using torch"""
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from brevitas import config
from model import CNV
from torch.backends import cudnn
from torch.utils import data as torch_data
from torchvision import transforms
from tqdm import tqdm

DATE_FORMAT = "%Y_%m_%d_%H_%M_%S"


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    config.IGNORE_MISSING_KEYS = True
    g = torch.Generator()
    g.manual_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)
    cudnn.deterministic = True

    batch_size = 4

    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    try:
        test_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=False,
            download=False,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    except RuntimeError:
        test_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=False,
            download=True,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )

    testloader = torch_data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    nb_steps = len(test_set) // batch_size

    checkpoint_path = Path(__file__).parent
    model_path = checkpoint_path / "8_bit_model.pt"
    loaded = torch.load(model_path)

    net = CNV(
        num_classes=len(classes), weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3
    )
    net.load_state_dict(loaded["model_state_dict"])
    net.eval()

    prediction_file = checkpoint_path / "predictions.csv"
    with open(prediction_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([f"{elt}_prob" for elt in classes] + ["label"])

        for _, data in (p_bar := tqdm(enumerate(testloader, 0), leave=False, total=nb_steps)):
            p_bar.set_description("Inference")

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            with torch.no_grad():
                # forward + backward + optimize
                outputs = net(inputs)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
            for preds, label in zip(outputs, labels):
                csv_writer.writerow(preds.numpy().tolist() + [label.numpy().tolist()])

    print("Finished inference")


if __name__ == "__main__":
    main()
