#!/usr/bin/env python
# coding: utf-8

# Run p_error search for CIFAR-10 8-bits network.

# Steps:
# 1. Load CIFAR-10 8-bits model

# 2. Load CIFAR-10 data-set

# 3. Pre-process the data-set, in our case, that means:
#   - Reducing the data-set size (that we will can calibration data in our experiments)
#   - Computing the features maps of the model on the client side

# 4. Run the search for a given set of hyper-parameters
#   - The objective is to look for the largest `p_error = i`, with i ∈ ]0,0.9[ ∩ ℝ,
#       which gives a model_i that has `accuracy_i`, such that:
#       | accuracy_i - accuracy_0| <= Threshold, where:
#           - Threshold is given by the user and
#           - `accuracy_0` refers to original model with `p_error ~ 0`

#   - Define our objective:
#       - If the objective is matched -> update the lower bound to be the current p-error
#       - Else, update the upper bound to be the current p-error
#   - Update the current p-error with the mean of the bounds

#   - The search terminates once it reaches the maximum number of iterations

#   - The inference is performed via the FHE simulation mode

# `p_error` is bounded between 0 and 0.9
#       - `p_error ~ 0.0`, refers to the original model in clear, that gives an accuracy
#           that we note as `accuracy_0`
#       - By default, `lower = 0.0` and `uppder` bound to 0.9.

#   - Run the inference in FHE simulation mode
#   - Define our objective:
#       - If the objective is matched -> update the lower bound to be the current p-error
#       - Else, update the upper bound to be the current p-error
#   - Update the current p-error with the mean of the bounds

import argparse
from typing import Dict, Optional

import numpy
import torch
from models import cnv_2w2a
from sklearn.metrics import top_k_accuracy_score
from torchvision import datasets, transforms

from concrete.ml.search_parameters import BinarySearch


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


DATASETS_ARGS = {
    "CIFAR10": {
        "dataset": datasets.CIFAR10,
        "train_transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "test_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    }
}


MODELS_ARGS = {
    "CIFAR10_8b": {
        "model_class": cnv_2w2a,
        "path": "experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar",
        "params": {
            "num_classes": 10,
            "weight_bit_width": 2,
            "act_bit_width": 2,
            "in_bit_width": 3,
            "in_ch": 3,
        },
    },
}


def main(args):

    if args.verbose:
        print(f"** Download `{args.dataset_name=}` dataset")
    dataset = get_torchvision_dataset(
        DATASETS_ARGS[args.dataset_name], train_set=True, max_examples=args.n_sample
    )

    all_y, all_x = [], []
    # Iterate over n batches, to apply any necessary torch transformations to the data-set
    for x_batch, y_batch in dataset:
        all_x.append(numpy.expand_dims(x_batch.numpy(), 0))
        all_y.append(y_batch)
    x_calib, y = numpy.concatenate(all_x), numpy.asarray(all_y)

    if args.verbose:
        print(f"** Load `{args.model_name=}` network")

    checkpoint = torch.load(MODELS_ARGS[args.model_name]["path"], map_location=args.device)
    model = cnv_2w2a(pre_trained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    if args.verbose:
        print("** `p_error` search")

    search = BinarySearch(
        estimator=model, predict="predict", metric=top_k_accuracy_score, verbose=args.verbose
    )

    p_error = search.run(x=x_calib, ground_truth=y, strategy=all)

    if args.verbose:
        print(f"Optimal p_error: {p_error}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--n_sample", type=int, default=500, help="n_sample")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10"],
        help="The selected dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CIFAR10_8b",
        choices=["CIFAR10_8b"],
        help="The selected model",
    )

    args = parser.parse_args()

    # Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed. For now, we
    # observe a decrease in torch's top1 accuracy when using MPS devices
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3953
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    main(args)
