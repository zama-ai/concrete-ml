"""Serialization module."""
import os

from torch.nn.modules import activation

# If the use of Skops needs to be disabled.
USE_SKOPS = bool(os.environ.get("USE_SKOPS", 1))

# Define all currently supported Torch activation functions
SUPPORTED_TORCH_ACTIVATIONS = [
    activation.CELU,
    activation.ELU,
    activation.GELU,
    activation.Hardshrink,
    activation.Hardsigmoid,
    activation.Hardswish,
    activation.Hardtanh,
    activation.LeakyReLU,
    activation.LogSigmoid,
    activation.LogSoftmax,
    activation.Mish,
    activation.PReLU,
    activation.ReLU,
    activation.ReLU6,
    activation.SELU,
    activation.SiLU,
    activation.Sigmoid,
    activation.Softmin,
    activation.Softplus,
    activation.Softshrink,
    activation.Softsign,
    activation.Tanh,
    activation.Tanhshrink,
    activation.Threshold,
]

# Some Torch activation functions are currently not supported in Concrete ML
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/335
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3518
UNSUPPORTED_TORCH_ACTIVATIONS = [
    activation.GLU,
    activation.MultiheadAttention,
    activation.RReLU,
    activation.Softmax,
    activation.Softmax2d,
]
