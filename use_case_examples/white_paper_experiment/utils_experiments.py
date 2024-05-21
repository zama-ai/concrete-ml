import pickle as pkl
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from brevitas import config
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# Normalization parameters for MNIST data
MEAN = STD = 0.5


def plot_dataset(data_loader: DataLoader, n: int = 10) -> None:
    """Visualize some images from a given data loader.

    Args:
        data_loader (DataLoader): Data loader.
        n (int): Limit the number of images to display to `n`.
    """

    # Class names
    class_names = data_loader.dataset.classes

    _, ax = plt.subplots(figsize=(12, 6))

    images, labels = next(iter(data_loader))

    # Make a grid from batch and rotate to get a valid shape for imshow
    images = make_grid(images[:n], nrow=n).permute((1, 2, 0))
    # Remove the previous normalization
    images = images * np.array(STD) + np.array(MEAN)

    ax.imshow(images)

    ax.set_title("".join([f"{class_names[img]:<13}" for img in labels[:n]]))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def mapping_keys(pre_trained_weights: Dict, model: torch.nn.Module, device: str) -> torch.nn.Module:
    """
    Initialize the quantized model with pre-trained fp32 weights.

    Args:
        pre_trained_weights (Dict): The state_dict of the pre-trained fp32 model.
        model (nn.Module): The Brevitas model.
        device (str): Device type.

    Returns:
        Callable: The quantized model with the pre-trained state_dict.
    """

    # Brevitas requirement to ignore missing keys
    config.IGNORE_MISSING_KEYS = True

    old_keys = list(pre_trained_weights.keys())
    new_keys = list(model.state_dict().keys())
    new_state_dict = OrderedDict()

    for old_key, new_key in zip(old_keys, new_keys):
        new_state_dict[new_key] = pre_trained_weights[old_key]

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    return model


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    param: Dict,
    step: int = 1,
    device: str = "cpu",
) -> torch.nn.Module:
    """Training the model.

    Args:
        model (nn.Module): A PyTorch or Brevitas network.
        train_loader (DataLoader): The training set.
        test_loader (DataLoader): The test set.
        param (Dict): Set of hyper-parameters to use depending on whether
            CIFAR-10 or CIFAR-100 is used.
        step (int): Display the loss and accuracy every `epoch % step`.
        device (str): Device type.
    Returns:
        nn.Module: the trained model.
    """

    param["accuracy_test"] = param.get("accuracy_test", [])
    param["accuracy_train"] = param.get("accuracy_train", [])
    param["loss_test_history"] = param.get("loss_test_history", [])
    param["loss_train_history"] = param.get("loss_train_history", [])
    param["criterion"] = param.get("criterion", torch.nn.CrossEntropyLoss())

    if param["seed"]:

        torch.manual_seed(param["seed"])
        random.seed(param["seed"])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=param["milestones"], gamma=param["gamma"]
    )

    # Save the state_dict
    dir = Path(".") / param["dir"] / param["training"]
    dir.mkdir(parents=True, exist_ok=True)

    # To avoid breaking up the tqdm bar
    with tqdm(total=param["epochs"], file=sys.stdout) as pbar:

        for i in range(param["epochs"]):
            # Train the model
            model.train()
            loss_batch_train, accuracy_batch_train = [], []

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                yhat = model(x)
                loss_train = param["criterion"](yhat, y)
                loss_train.backward()
                optimizer.step()

                loss_batch_train.append(loss_train.item())
                accuracy_batch_train.extend((yhat.argmax(1) == y).cpu().float().tolist())

            if scheduler:
                scheduler.step()

            param["accuracy_train"].append(np.mean(accuracy_batch_train))
            param["loss_train_history"].append(np.mean(loss_batch_train))

            # Evaluation during training:
            # Disable autograd engine (no backpropagation)
            # To reduce memory usage and to speed up computations
            with torch.no_grad():
                # Notify batchnormalization & dropout layers to work in eval mode
                model.eval()
                loss_batch_test, accuracy_batch_test = [], []
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    yhat = model(x)
                    loss_test = param["criterion"](yhat, y)
                    loss_batch_test.append(loss_test.item())
                    accuracy_batch_test.extend((yhat.argmax(1) == y).cpu().float().tolist())

                param["accuracy_test"].append(np.mean(accuracy_batch_test))
                param["loss_test_history"].append(np.mean(loss_batch_test))

            if i % step == 0:
                pbar.write(
                    f"Epoch {i:2}: Train loss = {param['loss_train_history'][-1]:.4f} "
                    f"VS Test loss = {param['loss_test_history'][-1]:.4f} - "
                    f"Accuracy train: {param['accuracy_train'][-1]:.4f} "
                    f"VS Accuracy test: {param['accuracy_test'][-1]:.4f}"
                )
                pbar.update(step)

    print("Save in:", f"{dir}/{param['dataset_name']}_{param['training']}_state_dict.pt")
    torch.save(
        model.state_dict(), f"{dir}/{param['dataset_name']}_{param['training']}_state_dict.pt"
    )

    with open(f"{dir}/{param['dataset_name']}_history.pkl", "wb") as f:
        pkl.dump(param, f)

    torch.cuda.empty_cache()

    return model


def torch_inference(
    model: torch.nn.Module,
    data: DataLoader,
    device: str = "cpu",
    verbose: bool = False,
) -> float:
    """Returns the `top_k` accuracy.

    Args:
        model (torch.nn.Module): A PyTorch or Brevitas network.
        data (DataLoader): The test or evaluation set.
        device (str): Device type.
        verbose (bool): For display.
    Returns:
        float: The top_k accuracy.
    """
    correct = []
    total_example = 0
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for x, y in tqdm(data, disable=verbose is False):
            x, y = x.to(device), y
            yhat = model(x).cpu().detach()
            correct.append(yhat.argmax(1) == y)
            total_example += len(x)

    return np.mean(np.vstack(correct), dtype="float64")


def format_results_df(PAPER_NOTES, results_cml, prefix):
    return pd.DataFrame(
        [
            [
                20,
                PAPER_NOTES[20][1],
                PAPER_NOTES[20][0],
                results_cml[20][0],
                results_cml[20][1],
                results_cml[20][2],
                PAPER_NOTES[20][0] / results_cml[20][2],
            ],
            [
                50,
                PAPER_NOTES[50][1],
                PAPER_NOTES[50][0],
                results_cml[50][0],
                results_cml[50][1],
                results_cml[50][2],
                PAPER_NOTES[50][0] / results_cml[50][2],
            ],
        ],
        columns=[
            "Num Layers",
            "Accuracy [1]",
            "FHE Latency [1]",
            f"{prefix} Accuracy fp32",
            f"{prefix} Accuracy FHE",
            f"{prefix} FHE Latency",
            "Speedup",
        ],
    ), {
        "Accuracy [1]": "{:,.1%}".format,
        "FHE Latency [1]": "{:,.2f}s".format,
        f"{prefix} Accuracy fp32": "{:,.1%}".format,
        f"{prefix} Accuracy FHE": "{:,.1%}".format,
        f"{prefix} FHE Latency": "{:,.2f}s".format,
        "Speedup": "{:,.1f}x".format,
    }
