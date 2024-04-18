import pickle as pkl
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from brevitas import config
from concrete.fhe.compilation import Configuration
from models import Fp32VGG11
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from concrete.ml.torch.compile import compile_brevitas_qat_model

warnings.filterwarnings("ignore", category=UserWarning)


DATASETS_ARGS = {
    "CIFAR_10": {
        "dataset": datasets.CIFAR10,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.247, 0.243, 0.261],
        "train_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                # We apply data augmentation in order to prevent overfitting
                transforms.RandomRotation(5, fill=(1,)),
                transforms.GaussianBlur(kernel_size=(3, 3)),
                transforms.RandomHorizontalFlip(0.5),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    },
    "CIFAR_100": {
        "dataset": datasets.CIFAR100,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "train_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # We apply data augmentation in order to prevent overfitting
                transforms.RandomRotation(5, fill=(1,)),
                transforms.GaussianBlur(kernel_size=(3, 3)),
                transforms.RandomHorizontalFlip(0.5),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    },
    # Separate FMNIST from CIFAR directory
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3552
    "FashionMNIST": {
        "dataset": datasets.FashionMNIST,
        "mean": (0.2859),
        "std": (0.3530),
        "train_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2859), (0.3530)),
                # We apply data augmentation in order to prevent overfitting
                transforms.RandomRotation(5, fill=(1,)),
                transforms.GaussianBlur(kernel_size=(3, 3)),
                transforms.RandomHorizontalFlip(0.5),
            ]
        ),
        "test_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2859), (0.3530)),
            ]
        ),
    },
}


def get_torchvision_dataset(
    dataset_config: Dict,
    train_set: bool,
    max_examples: Optional[int] = None,
):
    """Get train or testing data-set.

    Args:
        param (Dict): Set of hyper-parameters to use based on the selected torchvision data-set.
            It must contain: data-set transformations (torchvision.transforms.Compose), and the
            data-set_size (Optional[int]).
        train_set (bool): Use train data-set if True, else testing data-set

    Returns:
        A torchvision data-sets.
    """

    transform = dataset_config["train_transform"] if train_set else dataset_config["test_transform"]
    dataset = dataset_config["dataset"](
        download=True, root="./data", train=train_set, transform=transform
    )

    if max_examples is not None:
        assert len(dataset) > max_examples, "Invalid max number of examples"
        dataset = torch.utils.data.random_split(
            dataset,
            [max_examples, len(dataset) - max_examples],
        )[0]

    return dataset


def get_dataloader(
    param: Dict,
) -> Tuple[DataLoader, DataLoader]:
    """Returns the training and the test loaders of either CIFAR-10 or CIFAR-100 data-set.

    The CIFAR data-set contains of `32*32` colored images.

    Args:
        param (Dict): Set of hyper-parameters to use depending on whether
           CIFAR-10 or CIFAR-100 is used.
    Return:
        Tuple[DataLoader, DataLoader]: Training and test data loaders.
    """

    g = torch.Generator()
    g.manual_seed(param["seed"])
    np.random.seed(param["seed"])
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])

    max_examples = param.get("dataset_size", None)
    train_dataset = get_torchvision_dataset(
        DATASETS_ARGS[param["dataset_name"]], train_set=True, max_examples=max_examples
    )

    test_dataset = get_torchvision_dataset(
        DATASETS_ARGS[param["dataset_name"]], train_set=False, max_examples=max_examples
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=param["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=g,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=param["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    return (train_loader, test_loader)


def plot_history(param: Dict, load: bool = False) -> None:
    """Display the loss and accuracy for the test and training sets.

    Args:
        param (Dict): Set of hyper-parameters to use depending on whether
            CIFAR-10 or CIFAR-100 is used.
        load (bool): If True, we upload the stored param.
    Returns:
        None.
    """

    if load:
        with open(
            f"{param['dir']}/{param['training']}/{param['dataset_name']}_history.pkl", "br"
        ) as f:
            param = pkl.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Plot the training and test loss.
    axes[0].plot(
        range(len(param["loss_train_history"])),
        param["loss_train_history"],
        label="Train loss",
        marker="o",
    )
    axes[0].plot(
        range(len(param["loss_test_history"])),
        param["loss_test_history"],
        label="Test loss",
        marker="s",
    )

    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    # Plot the training and test accuracy.
    axes[1].plot(
        range(len(param["accuracy_train"])),
        param["accuracy_train"],
        label="Train accuracy",
        marker="o",
    )
    axes[1].plot(
        range(len(param["accuracy_test"])),
        param["accuracy_test"],
        label="Test accuracy",
        marker="s",
    )

    axes[1].set_title("Top-1 accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


def plot_baseline(param: Dict, data: DataLoader, device: str) -> None:
    """
    Display the test accuracy given `param` arguments
    that we got using Transfer Learning and QAT approaches.

    Args:
        param (Dict): Set of hyper-parameters to use depending on whether
            CIFAR-10 or CIFAR-100 is used.
        data (DataLoader): Test set.
        device (str): Device type.

    Returns:
        None
    """
    # The accuracy of the counterpart pre-trained model in fp32 will be used as a baseline.
    # That we try to catch up during the Quantization Aware Training.
    checkpoint = torch.load(f"{param['dir']}/{param['pre_trained_path']}", map_location=device)
    fp32_vgg = Fp32VGG11(param["output_size"])
    fp32_vgg.load_state_dict(checkpoint)
    baseline = torch_inference(fp32_vgg, data, device)

    plt.plot(
        range(len(param["accuracy_test"])),
        param["accuracy_test"],
        marker="o",
        label="Test accuracy",
    )
    plt.text(x=0, y=baseline + 0.01, s=f"Baseline = {baseline * 100: 2.2f}%", fontsize=15, c="red")
    plt.plot(range(len(param["accuracy_test"])), [baseline] * len(param["accuracy_test"]), "r--")

    plt.title(f"Accuracy on the testing set with {param['dataset_name']}")
    plt.legend(loc="best")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.xlim(-0.3, 4.2)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def plot_dataset(data_loader: DataLoader, param: Dict, n: int = 10) -> None:
    """Visualize some images from a given data loader.

    Args:
        data_loader (DataLoader): Data loader.
        param (Dict): Set of hyper-parameters to use depending on whether
            CIFAR-10 or CIFAR-100 is used.
        n (int): Limit the number of images to display to `n`.
    Returns:
        None
    """

    # Class names
    class_names = data_loader.dataset.classes

    _, ax = plt.subplots(figsize=(12, 6))

    images, labels = next(iter(data_loader))

    # Make a grid from batch and rotate to get a valid shape for imshow
    images = make_grid(images[:n], nrow=n).permute((1, 2, 0))
    # Remove the previous normalization
    images = images * np.array(DATASETS_ARGS[param["dataset_name"]]["std"]) + np.array(
        DATASETS_ARGS[param["dataset_name"]]["mean"]
    )

    ax.imshow(images)

    ax.set_title("".join([f"{class_names[img]:<13}" for img in labels[:n]]))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    param: Dict,
    step: int = 1,
    device: str = "cpu",
) -> nn.Module:
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

    torch.manual_seed(param["seed"])
    random.seed(param["seed"])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=param["milestones"], gamma=param["gamma"]
    )

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

    # Save the state_dict
    dir = Path(".") / param["dir"] / param["training"]
    dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(), f"{dir}/{param['dataset_name']}_{param['training']}_state_dict.pt"
    )

    with open(f"{dir}/{param['dataset_name']}_history.pkl", "wb") as f:
        pkl.dump(param, f)

    torch.cuda.empty_cache()

    return model


def torch_inference(
    model: nn.Module,
    data: DataLoader,
    device: str = "cpu",
    verbose: bool = False,
) -> float:
    """Returns the `top_k` accuracy.

    Args:
        model (nn.Module): A PyTorch or Brevitas network.
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


def fhe_compatibility(model: Callable, data: DataLoader) -> Callable:
    """Test if the model is FHE-compatible.

    Args:
        model (Callable): The Brevitas model.
        data (DataLoader): The data loader.

    Returns:
        Callable: Quantized model.
    """

    qmodel = compile_brevitas_qat_model(
        model.to("cpu"),
        # Training
        torch_inputset=data,
        show_mlir=False,
        output_onnx_file="test.onnx",
    )

    return qmodel


def mapping_keys(pre_trained_weights: Dict, model: nn.Module, device: str) -> nn.Module:
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


def fhe_simulation_inference(quantized_module, data_loader, verbose: bool = False) -> float:
    """Evaluate the model in FHE simulation mode.

    Args:
        quantized_module (Callable): The quantized module.
        data_loader (int): The test or evaluation set.
        verbose (bool): For display.

    Returns:
        float: The accuracy measured through FHE simulation
    """
    correct_sim = []
    total_example = 0

    disable_tqdm = not verbose
    for data, labels in tqdm(data_loader, disable=disable_tqdm):

        data, labels = data.detach().cpu().numpy(), labels.detach().cpu().numpy()

        # Store the predicted quantized probabilities
        predictions = quantized_module.forward(data, fhe="simulate")

        total_example += data.shape[0]

        # Store the class predictions
        correct_sim.extend(predictions.argmax(1) == labels)

    acc = np.mean(correct_sim, dtype="float64")
    return acc
