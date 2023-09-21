# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Original file can be found at https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/logger.py
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.optim as optim
from logger import EvalEpochMeters, Logger, TrainingEpochMeters
from models import model_with_cfg
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[torch.FloatTensor]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    In top-5 accuracy you give yourself credit for having the right answer if the right answer
    appears in your top five guesses.
    Taken from https://discuss.pytorch.org/t/top-k-error-calculation/48815/2.

    Args:
        output (torch.Tensor): The prediction of the model (scores, logits, raw y_pred) before
            normalization or getting classes
        target (torch.Tensor) : The target data representing the truth.
        topk (Tuple[int]): Tuple of topk's to compute top 1, top 2 and top 5.

    Returns:
      list_topk_accs (List[torch.FloatTensor]): The list of topk accuracies [top1st, top2nd, ...]
        depending on the topk input.
    """
    with torch.no_grad():
        # Get the largest k and batch size
        max_k = max(topk)
        batch_size = target.size(0)

        # Get top max_k indices that correspond to the most likely probability scores
        _, y_pred = output.topk(k=max_k, dim=1)

        # Transpose it to shape [max_k, batch_size]
        y_pred = y_pred.t()

        # Get the credit for each example if the models predictions is in max_k values
        target_reshaped = target.view(1, -1).expand_as(y_pred)

        # Compare every topk's model prediction with the ground truth
        correct = y_pred == target_reshaped

        # Compute topk accuracies
        list_topk_accs = []
        for k in topk:
            # Get topk answers as a float tensor (1.0 and 0.0)
            # The clone is necessary here in order to avoid miss-representations
            topk_matched_truth = correct[:k].reshape(-1).clone().float()

            # Get the total of right matches
            total_correct_topk = topk_matched_truth.sum(dim=0, keepdim=True)

            # Compute and store the topk accuracy for this batch (and k)
            topk_acc = total_correct_topk / batch_size
            list_topk_accs.append(topk_acc)

        return list_topk_accs


def get_train_set(dataset: str, datadir: Union[str, Path]) -> Dataset:
    """Retrieve a dataset's training set.

    Args:
        dataset (str): The dataset's name.
        datadir (Union[str, Path]): The dataset's directory path to consider.

    Returns:
        train_set (Dataset): The training set.
    """
    if dataset == "CIFAR10":

        # Transform pipeline that normalizes the data between -1 and 1
        transformations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
        transform_train = transforms.Compose(transformations)
        builder = CIFAR10
    else:
        raise Exception(f"Dataset not supported: {dataset}")

    train_set = builder(root=datadir, train=True, download=True, transform=transform_train)

    return train_set


def get_test_set(dataset: str, datadir: Union[str, Path]) -> Dataset:
    """Retrieve a dataset's test set.

    Args:
        dataset (str): The dataset's name.
        datadir (Union[str, Path]): The dataset's directory path to consider.

    Returns:
        train_set (Dataset): The test set.
    """
    if dataset == "CIFAR10":

        # Transform pipeline that normalizes the data between -1 and 1
        transformations = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
        transform_test = transforms.Compose(transformations)
        builder = CIFAR10
    else:
        raise Exception(f"Dataset not supported: {dataset}")

    test_set = builder(root=datadir, train=False, download=True, transform=transform_test)
    return test_set


class Trainer(object):
    def __init__(self, args):

        model, cfg = model_with_cfg(args.network, args.pre_trained)

        # Init arguments
        self.args = args
        prec_name = "_{}W{}A".format(
            cfg.getint("QUANT", "WEIGHT_BIT_WIDTH"), cfg.getint("QUANT", "ACT_BIT_WIDTH")
        )
        experiment_name = "{}{}_{}".format(
            args.network, prec_name, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.output_dir_path = os.path.join(args.experiments, experiment_name)

        if self.args.resume:
            self.output_dir_path, _ = os.path.split(args.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)

        if not args.dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, "checkpoints")
            if not args.resume:
                os.mkdir(self.output_dir_path)
                os.mkdir(self.checkpoints_dir_path)
        self.logger = Logger(self.output_dir_path, args.dry_run)

        # Randomness
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        dataset = cfg.get("MODEL", "DATASET")

        train_set = get_train_set(dataset, args.datadir)
        test_set = get_test_set(dataset, args.datadir)

        self.train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        self.test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        # Setup device
        if args.gpus is not None:
            args.gpus = [int(i) for i in args.gpus.split(",")]
            self.device = "cuda:" + str(args.gpus[0])
            torch.backends.cudnn.benchmark = True
        else:
            # Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed. For
            # now, we observe a decrease in torch's top1 accuracy when using MPS devices
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3953
            self.device = "cpu"

        self.device = torch.device(self.device)

        # Resume checkpoint, if any
        if args.resume:
            print("Loading model checkpoint at: {}".format(args.resume))
            package = torch.load(args.resume, map_location="cpu")
            model_state_dict = package["state_dict"]
            model.load_state_dict(model_state_dict, strict=args.strict)

        if args.gpus is not None and len(args.gpus) > 1:
            model = nn.DataParallel(model, args.gpus)
        else:
            model = model.to(device=self.device)

        self.model = model

        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device=self.device)

        # Init optimizer
        if args.optim == "ADAM":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optim == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        # Resume optimizer, if any
        if args.resume and not args.evaluate:
            self.logger.log.info("Loading optimizer checkpoint")
            if "optim_dict" in package.keys():
                self.optimizer.load_state_dict(package["optim_dict"])
            if "epoch" in package.keys():
                self.starting_epoch = package["epoch"]
            if "best_val_acc" in package.keys():
                self.best_val_acc = package["best_val_acc"]

        # LR scheduler
        if args.scheduler == "STEP":
            milestones = [int(i) for i in args.milestones.split(",")]
            self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=0.1)
        elif args.scheduler == "FIXED":
            self.scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(self.args.scheduler))

        # Resume scheduler, if any
        if args.resume and not args.evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package["epoch"] - 1

    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optim_dict": self.optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": self.best_val_acc,
            },
            best_path,
        )

    def train_model(self):

        # training starts
        if self.args.detect_nan:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.starting_epoch, self.args.epochs):

            # Set to training mode
            self.model.train()
            self.criterion.train()

            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()

            for i, data in enumerate(self.train_loader):
                (input, target) = data
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                target_var = target

                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                # Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target_var)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.model.clip_weights(-1, 1)

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(
                        epoch_meters, epoch, i, len(self.train_loader)
                    )

                # training batch ends
                start_data_loading = time.time()

            # Set the learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            else:
                # Set the learning rate
                if epoch % 40 == 0:
                    self.optimizer.param_groups[0]["lr"] *= 0.5

            # Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # checkpoint
            if top1avg >= self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch, "best.tar")
            elif not self.args.dry_run:
                self.checkpoint_best(epoch, "checkpoint.tar")

        # training ends
        if not self.args.dry_run:
            return os.path.join(self.checkpoints_dir_path, "best.tar")

    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            # compute loss
            loss = self.criterion(output, target)
            eval_meters.loss_time.update(time.time() - end)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            # Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        return eval_meters.top1.avg
