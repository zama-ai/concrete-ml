"""Custom Quantiation Aware Training Brevitas quantizers."""
from brevitas.quant.scaled_int import (
    IntQuant,
    MaxStatsScaling,
    ParamFromRuntimePercentileScaling,
    PerTensorPoTScaling8bit,
    WeightQuantSolver,
)
from brevitas.quant.solver.act import ActQuantSolver


# pylint: disable-next=too-many-ancestors
class Int8ActPerTensorPoT(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorPoTScaling8bit, ActQuantSolver
):
    """Quantization options for power-of-two scaling activations."""

    _partialmethod = None


# pylint: disable-next=too-many-ancestors
class Int8WeightPerTensorPoT(IntQuant, MaxStatsScaling, PerTensorPoTScaling8bit, WeightQuantSolver):
    """Quantization options for power-of-two scaling weights."""

    _partialmethod = None
