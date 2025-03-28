"""This module contains classes for LoRA (Low-Rank Adaptation) FHE training and custom layers."""

import copy
import logging
from collections import UserDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from ..common.utils import assert_true
from .hybrid_backprop_linear import CustomLinear
from .hybrid_model import HybridFHEModel

try:
    from transformers import Conv1D as TransformerConv1D
except ImportError:  # pragma: no cover
    TransformerConv1D = None

# Create a tuple of linear layer classes to check against
LINEAR_LAYERS: tuple = (nn.Linear,)
if TransformerConv1D is not None:
    LINEAR_LAYERS = LINEAR_LAYERS + (TransformerConv1D,)


# pylint: disable=abstract-method
# pylint: disable=arguments-differ


def setup_logger(log_file: str, level=logging.INFO):
    """Set up a logger that logs to both console and a file.

    Args:
        log_file (str): The path to the log file.
        level (int): The logging level.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# pylint: disable=protected-access
def grad_to(param, device: str) -> None:
    """Move parameter gradient to device.

    Args:
        param: torch parameter with gradient
        device (str): target device for gradient
    """
    if param._grad is not None:
        param._grad.data = param._grad.data.to(device)  # pragma: no cover


def optimizer_to(optim, device):
    """Move optimizer object to device.

    Args:
        optim: torch optimizer
        device (str): target device for gradient
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        params_to_move = [param] if isinstance(param, torch.Tensor) else list(param.values())

        for subparam in params_to_move:
            if isinstance(subparam, torch.Tensor):
                subparam.data = subparam.data.to(device)
                grad_to(subparam, device)


class LoraTraining(torch.nn.Module):
    """LoraTraining module for fine-tuning with LoRA in a hybrid model setting.

    This class is designed to enable Low-Rank Adaptation (LoRA) fine-tuning
    in a hybrid model context. It allows selective execution of forward and
    backward passes in FHE.

    The class replaces standard linear layers with custom layers that are
    compatible with LoRA and FHE operations.

    Args:
        model (torch.nn.Module): The base model with LoRA layers to be fine-tuned.
        n_layers_to_skip_for_backprop (int): Number of initial linear layers to keep as standard
            layers. Since the first layer doesn't need backpropagation (no previous layer to
            update), we typically skip 1 layer. Defaults to 1.
        loss_fn (callable, optional): Loss function to compute the loss. If None, the model
            is expected to return a loss.
    """

    def __init__(self, model, n_layers_to_skip_for_backprop=1, loss_fn=None):
        super().__init__()

        # Check if model accepts labels when no loss_fn is provided
        if loss_fn is None:
            from inspect import signature

            forward_sig = signature(model.forward)
            if "labels" not in forward_sig.parameters:
                raise ValueError(
                    "When no loss_fn is provided, the model's forward method"
                    "must accept a 'labels' parameter"
                )

        # Assert that the model contains LoRA layers
        self.assert_has_lora_layers(model)

        self.inference_model = model
        self.replace_layers_with_custom(self.inference_model, n_layers_to_skip_for_backprop)

        self.calibrate = False
        self.loss_fn = loss_fn
        self.loss_scaling_factor = 1.0

    def set_loss_scaling_factor(self, loss_scaling_factor: float):
        """Set a scaling factor for the loss to account for gradient accumulation.

        This ensures that gradients are correctly averaged over multiple
        mini-batches when performing gradient accumulation, preventing them
        from being scaled up by the number of accumulation steps.

        Args:
            loss_scaling_factor (float): The number of gradient accumulation steps.
                                        The loss will be divided by this factor
                                        before backpropagation.
        """
        self.loss_scaling_factor = loss_scaling_factor

    @staticmethod
    def assert_has_lora_layers(model):
        """Assert that the model contains LoRA layers.

        Args:
            model (torch.nn.Module): The model to check for LoRA layers.

        Raises:
            ValueError: If the model does not contain any LoRA layers.
        """

        def is_lora_module(module):
            # Check for common LoRA attributes with case-insensitive matching
            lora_attributes = ["lora_a", "lora_b", "lora_dropout"]
            return any(
                hasattr(module, attr)
                or hasattr(module, attr.lower())
                or hasattr(module, attr.upper())
                for attr in lora_attributes
            )

        has_lora = any(is_lora_module(module) for module in model.modules())

        if not has_lora:
            raise ValueError("The model does not contain any detectable LoRA layers.")

        print("LoRA layers detected in the model.")

    @staticmethod
    def replace_layers_with_custom(model: nn.Module, n_layers_to_skip_for_backprop: int) -> None:
        """Replace linear layers with custom ones.

        Args:
            model (nn.Module): The model to replace layers in.
            n_layers_to_skip_for_backprop (int): Number of initial linear layers to keep as standard
                layers. Since the first layer doesn't need backpropagation (no previous layer to
                update), we typically skip 1 layer.
        """

        def _replace(module: nn.Module):
            nonlocal n_layers_to_skip_for_backprop
            for name, child in list(module.named_children()):

                # Skip lora layers as they are computed on the client side
                if "lora" in name:
                    continue

                if isinstance(child, LINEAR_LAYERS):
                    if n_layers_to_skip_for_backprop > 0:
                        n_layers_to_skip_for_backprop -= 1

                        # Skip the first eligible layer
                        continue

                    # Determine if weights need to be transposed
                    weight_transposed = TransformerConv1D is not None and isinstance(
                        child, TransformerConv1D
                    )

                    # Create the CustomLinear layer
                    custom_layer = CustomLinear(
                        weight=child.weight,
                        bias=child.bias,
                        weight_transposed=weight_transposed,
                    )

                    # Replace the original layer with the custom layer
                    setattr(module, name, custom_layer)
                else:
                    # Recursively apply to child modules
                    _replace(child)

        _replace(model)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, Union[Tensor, None]]:
        """Forward pass of the LoRA training module.

        Args:
            inputs (tuple): A tuple containing the input tensors.

        Returns:
            A tuple containing the original (unscaled) loss and None.

        Raises:
            ValueError: If the model does not return a loss and no loss function is provided.
        """
        attention_mask, labels = self.process_inputs(inputs)

        # Validate attention mask
        assert attention_mask is None or torch.all(
            torch.logical_or(attention_mask == 0, attention_mask == 1)
        ), "Invalid attention mask provided. Attention mask should only contain 0s and 1s."

        # Pass inputs and labels to the model
        if isinstance(inputs, (dict, UserDict)):
            outputs = self.inference_model(**inputs)
        else:
            outputs = self.inference_model(*inputs)

        if self.loss_fn is None:
            # Check if outputs is a dict and retrieve the loss
            loss = (
                outputs.get("loss", None)
                if isinstance(outputs, dict)
                else getattr(outputs, "loss", None)
            )
            if loss is None:
                raise ValueError(
                    "The model did not return a loss.",
                    (
                        "Ensure that the 'labels' key is populated in the training data dict, "
                        "or provide a loss_fn."
                    ),
                )
        else:
            # If logits is a dict with 'logits' key, extract it
            assert_true(
                isinstance(outputs, torch.Tensor)
                or (isinstance(outputs, dict) and "logits" in outputs),
                (
                    "When a loss function is "
                    "specified in the LoraTrainer constructor, the "
                    "LORA module to be trained must return either a Tensor or a  "
                    "dictionary containing the key `logits` with a Tensor value"
                ),
            )

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = self.loss_fn(logits, labels)

        # Scale the loss for gradient accumulation
        scaled_loss = loss / self.loss_scaling_factor

        # We need to set requires grad to the loss manually because the inference model's last
        # step is the "lm_head" layer, which might be detached from the graph by the hybrid model
        scaled_loss.requires_grad_(True)
        scaled_loss.backward()

        # Return the original (unscaled) loss for logging
        return loss.detach(), None

    def process_inputs(self, inputs: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process training inputs such as labels and attention mask.

        Args:
            inputs: a dict, BatchEncoding or tuple containing training data

        Returns:
            res: tuple containing the attention mask and the labels
        """

        if isinstance(inputs, (dict, UserDict)):
            attention_mask = inputs.get("attention_mask", None)
            if self.loss_fn is None:
                labels = inputs.get("labels", None)
            else:
                labels = inputs.pop("labels", None)
        else:

            assert isinstance(inputs, tuple)
            assert len(inputs) == 2, (
                "Tuple inputs to LoraTraining must have two elements (data, labels). "
                "Attention masks are not yet supported with tuples, use a dictionary input"
            )

            # Unpack depending on how many inputs we have
            assert len(inputs) == 2
            _, labels = inputs
            attention_mask = None

        return attention_mask, labels


# pylint: disable=too-many-instance-attributes
class LoraTrainer:
    """Trainer class for LoRA fine-tuning with FHE support.

    This class handles:
    - Training loop
    - Periodic logging and evaluation
    - Loss tracking
    - Integration with hybrid FHE model

    Args:
        model (nn.Module): The base model with LoRA layers to be fine-tuned.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (callable): Loss function to compute the loss.
        lr_scheduler (optional): Learning rate scheduler.
        training_args (dict): Training arguments.
        n_layers_to_skip_for_backprop (int): Number of initial linear layers to keep as standard
            layers. Since the first layer doesn't need backpropagation (no previous layer to
            update), we typically skip 1 layer. Defaults to 1.
        eval_loader (DataLoader, optional): DataLoader for evaluation data.
        eval_metric_fn (callable, optional): Function(model, eval_loader) -> dict of metrics.
        logging_steps (int, optional): Log loss every N training steps. Defaults to 1.
        eval_steps (int, optional): Evaluate on eval set every N training steps. Defaults to 10.
        train_log_path (str, optional): Path to a log file for training. Defaults to "training.log".
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        lr_scheduler=None,
        training_args=None,
        n_layers_to_skip_for_backprop=1,
        eval_loader: Optional[DataLoader] = None,
        eval_metric_fn: Optional[Callable] = None,
        logging_steps: int = 1,
        eval_steps: int = 10,
        train_log_path: str = "training.log",
        checkpoint_dir: str = None,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_args = training_args or {}
        self.gradient_accumulation_steps = self.training_args.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = self.training_args.get("max_grad_norm", None)

        self.eval_loader = eval_loader
        self.eval_metric_fn = eval_metric_fn
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.training_losses: List[float] = []
        self.gradient_stats: Dict[str, List[float]] = {}

        # Set up logging
        self.logger = setup_logger(train_log_path)
        self.logger.info("=== Starting new training session ===")

        assert_true(
            training_args is None
            or "use_cpu" not in training_args
            or training_args["use_cpu"] is True,
            (
                "When specifying custom training_args for the LoraTrainer, you "
                "must set use_cpu=True. The Concrete ML LoraTrainer can be "
                "executed on GPU by setting device='cuda' in the `train` call"
            ),
        )

        # Create the LoraTraining module
        self.lora_training_module = LoraTraining(
            model, n_layers_to_skip_for_backprop=n_layers_to_skip_for_backprop, loss_fn=loss_fn
        )

        # Determine modules to be executed remotely
        self.remote_names = get_remote_names(self.lora_training_module)

        # Create the hybrid model
        self.hybrid_model = HybridFHEModel(
            self.lora_training_module, module_names=self.remote_names
        )

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def compile(self, inputset, n_bits=8, use_dynamic_quantization=True):
        """Compile the hybrid model with the given input set.

        Args:
            inputset (tuple): Input set for compilation.
            n_bits (int): Bit width for quantization.
            use_dynamic_quantization (bool): Whether to use dynamic quantization.

        Returns:
            Tuple[int, int]: The epoch and global step of the latest checkpoint if found,
                else (0, 0).
        """
        # Load latest checkpoint if checkpoint_dir exists
        epoch, global_step = 0, 0
        if self.checkpoint_dir is not None:
            checkpoint_files = sorted(Path(self.checkpoint_dir).glob("checkpoint_epoch_*.pth"))
            if checkpoint_files:
                latest_checkpoint = str(checkpoint_files[-1])
                epoch, global_step = self.load_checkpoint(latest_checkpoint)

        self.hybrid_model.compile_model(
            copy.deepcopy(inputset),
            n_bits=n_bits,
            use_dynamic_quantization=use_dynamic_quantization,
        )

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4707
        # Need a forward call to set the executors in remote modules
        self.hybrid_model.set_fhe_mode("disable")
        self.hybrid_model(inputset)
        self.logger.info("Compilation complete.")

        return epoch, global_step

    def _evaluate(self, step: int):
        if self.eval_loader and self.eval_metric_fn:
            self.logger.info("Running evaluation at step %d...", step)
            self.lora_training_module.inference_model.eval()
            metrics: Dict[str, float] = self.eval_metric_fn(
                self.lora_training_module.inference_model, self.eval_loader
            )
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info("[Evaluation at step %d] %s", step, metrics_str)
            self.lora_training_module.inference_model.train()  # back to train mode

        else:
            self.logger.info("No evaluation data or metric function provided.")

    def save_checkpoint(self, epoch: int, global_step: int):
        """Save a training checkpoint.

        Args:
            epoch (int): The current epoch number.
            global_step (int): The current global step number.
        """
        assert self.checkpoint_dir is not None, "Checkpoint directory is not set"
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
        save_dict = {
            "model_state_dict": self.lora_training_module.inference_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict() if self.lr_scheduler else None
            ),
            "training_losses": self.training_losses,
            "global_step": global_step,
            "epoch": epoch,
            "gradient_stats": self.gradient_stats,
        }
        torch.save(save_dict, checkpoint_path)
        self.logger.info("Checkpoint saved at %s", checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint and restore model, optimizer, and lr_scheduler.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            Tuple[int, int]: The epoch and global step of the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
        """
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.lora_training_module.inference_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler and checkpoint["lr_scheduler_state_dict"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.training_losses = checkpoint.get("training_losses", [])
        global_step = checkpoint.get("global_step", 0)
        epoch = checkpoint.get("epoch", 0)
        self.gradient_stats = checkpoint.get("gradient_stats", {})
        self.logger.info(
            "Checkpoint loaded from %s (Epoch %d, Step %d)", checkpoint_path, epoch, global_step
        )
        return epoch, global_step

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 1,
        fhe: str = "simulate",
        device: str = "cpu",
    ):
        """Train the model using the hybrid FHE model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            fhe (str): FHE mode ('disable', 'simulate', 'execute' or 'torch').
            device (str): A device string that is compatible with PyTorch, used for
                client-side computations.
        """
        self.hybrid_model.model.to(device)
        optimizer_to(self.optimizer, device)

        self.lora_training_module.inference_model.train()

        # Set the loss scaling factor for gradient accumulation
        self.lora_training_module.set_loss_scaling_factor(self.gradient_accumulation_steps)

        start_epoch = 1
        global_step = 0

        for epoch in range(start_epoch, num_epochs + 1):
            total_loss = 0.0
            steps_this_epoch = 0
            self.optimizer.zero_grad()

            epoch_bar = tqdm(
                enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", leave=False
            )
            for step, batch in epoch_bar:
                global_step += 1
                steps_this_epoch += 1
                if isinstance(batch, (UserDict, dict)):
                    # Convert dict to tuple of values and move them to the device
                    batch = {k: v.to(device) for k, v in batch.items()}
                else:
                    assert isinstance(batch, (tuple, list))
                    # Move tuple/list elements to the device
                    batch = tuple(
                        item.to(device) if isinstance(item, torch.Tensor) else item
                        for item in batch
                    )

                # Forward pass through the hybrid model
                loss, _ = self.hybrid_model(batch, fhe=fhe)

                # Loss scaling and backward is done inside LoraTraining

                # Accumulate loss for logging
                total_loss += loss.item()
                self.training_losses.append(loss.item())

                # Logging
                if global_step % self.logging_steps == 0:
                    avg_loss = total_loss / steps_this_epoch
                    self.logger.info(
                        "Step %d: loss=%f, avg_loss=%f",
                        global_step,
                        loss.item(),
                        avg_loss,
                    )
                    # Log gradients at the same time as loss
                    self._log_gradients()

                # Evaluation
                if global_step % self.eval_steps == 0:
                    self._evaluate(global_step)

                # Gradient accumulation steps
                if ((step + 1) % self.gradient_accumulation_steps == 0) or (
                    step + 1 == len(train_loader)
                ):
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.lora_training_module.parameters(), self.max_grad_norm
                        )

                    # Optimizer step
                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                epoch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = total_loss / len(train_loader)
            self.logger.info(
                "Epoch %d completed. Avg Loss: %f, FHE Mode: %s",
                epoch,
                avg_epoch_loss,
                fhe,
            )

            # Save checkpoint after each epoch
            if self.checkpoint_dir:
                self.save_checkpoint(epoch, global_step)

        self.logger.info(
            "Training completed. Final Avg Loss: %f, FHE Mode: %s",
            avg_epoch_loss,
            fhe,
        )

    def get_training_losses(self):
        """Return all recorded training losses.

        Returns:
            List[float]: All recorded training losses.
        """
        return self.training_losses

    def save_and_clear_private_info(self, path):
        """Save the model and remove private information.

        Args:
            path (str): The path to save the model.
        """
        self.hybrid_model.save_and_clear_private_info(path)
        self.logger.info("Model saved at %s", path)

    def _log_gradients(self):
        """Calculate and log gradient statistics for each layer."""
        grad_stats = {}
        for name, param in self.lora_training_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm

                # Store in history
                if name not in self.gradient_stats:
                    self.gradient_stats[name] = []
                self.gradient_stats[name].append(grad_norm)

        # Log average gradient magnitude across all layers
        if grad_stats:
            avg_grad = sum(grad_stats.values()) / len(grad_stats)
            self.logger.info("Average gradient magnitude: %f", avg_grad)

    def get_gradient_stats(self):
        """Return recorded gradient statistics.

        Returns:
            Dict[str, List[float]]: Gradient statistics per layer over time.
        """
        return self.gradient_stats


def get_remote_names(model: nn.Module, include_embedding_layers: bool = False) -> List[str]:
    """Get names of modules to be executed remotely.

    Args:
        model (nn.Module): The model to inspect.
        include_embedding_layers (bool): Whether to include embedding layers.

    Returns:
        List[str]: List of module names to be executed remotely.
    """
    remote_names = []
    for name, module in model.named_modules():
        # Skip if the name contains 'lora' since they will be done on client side
        if "lora" in name:
            continue

        # Skip 'lm_head' if embedding layers are not included
        is_lm_head = "lm_head" in name
        if is_lm_head and not include_embedding_layers:
            continue

        # Handle different module types
        if isinstance(module, LINEAR_LAYERS):
            remote_names.append(name)
        elif isinstance(module, CustomLinear):
            remote_names.append(f"{name}.forward_module")
            remote_names.append(f"{name}.backward_module")
        elif include_embedding_layers and (isinstance(module, nn.Embedding) or is_lm_head):
            remote_names.append(name)

    return remote_names
