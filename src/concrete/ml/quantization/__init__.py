"""Modules for quantization."""
from .post_training import PostTrainingAffineQuantization
from .quantized_array import QuantizedArray
from .quantized_module import QuantizedModule
from .quantized_ops import (
    QuantizedClip,
    QuantizedExp,
    QuantizedGemm,
    QuantizedLinear,
    QuantizedOp,
    QuantizedRelu,
    QuantizedSigmoid,
    QuantizedTanh,
)
