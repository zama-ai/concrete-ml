"""Modules for quantization."""
from .base_quantized_op import QuantizedOp
from .post_training import PostTrainingAffineQuantization
from .quantized_array import QuantizedArray
from .quantized_module import QuantizedModule
from .quantized_ops import (
    QuantizedAbs,
    QuantizedAdd,
    QuantizedAvgPool,
    QuantizedCelu,
    QuantizedClip,
    QuantizedConv,
    QuantizedElu,
    QuantizedExp,
    QuantizedGemm,
    QuantizedGreater,
    QuantizedHardSigmoid,
    QuantizedHardSwish,
    QuantizedIdentity,
    QuantizedLeakyRelu,
    QuantizedLinear,
    QuantizedLog,
    QuantizedMatMul,
    QuantizedMul,
    QuantizedPad,
    QuantizedPRelu,
    QuantizedRelu,
    QuantizedReshape,
    QuantizedSelu,
    QuantizedSigmoid,
    QuantizedSoftplus,
    QuantizedSub,
    QuantizedTanh,
    QuantizedWhere,
)
