# Brevitas
import brevitas.nn as qnn
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from brevitas import quant
from brevitas.core import bit_width
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.nn.quant_activation import QuantIdentity
from brevitas.quant import *
from brevitas.quant import IntBias
from brevitas.quant.base import *
from brevitas.quant.base import ParamFromRuntimeMinMaxScaling
from brevitas.quant.solver import *
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from dependencies import value
from torch.nn import init

# Concrete-Python
from concrete import fhe


class MNISTQATModel(nn.Module):
    def __init__(self, a_bits, w_bits):
        super(MNISTQATModel, self).__init__()

        self.a_bits = a_bits
        self.w_bits = w_bits

        self.cfg = [28 * 28, 192, 192, 192, 10]

        self.quant_inp = qnn.QuantIdentity(
            act_quant=CommonActQuant if a_bits is not None else None,
            bit_width=a_bits,
            return_quant_tensor=True,
        )

        self.fc1 = qnn.QuantLinear(
            self.cfg[0],
            self.cfg[1],
            False,
            weight_quant=CommonWeightQuant if w_bits is not None else None,
            weight_bit_width=w_bits,
            bias_quant=None,
        )

        self.bn1 = nn.BatchNorm1d(self.cfg[1], momentum=0.999)
        self.q1 = QuantIdentity(
            act_quant=CommonActQuant, bit_width=a_bits, return_quant_tensor=True
        )

        self.fc2 = qnn.QuantLinear(
            self.cfg[1],
            self.cfg[2],
            False,
            weight_quant=CommonWeightQuant if w_bits is not None else None,
            weight_bit_width=w_bits,
            bias_quant=None,  # FheBiasQuant if w_bits is not None else None,
        )

        self.bn2 = nn.BatchNorm1d(self.cfg[1], momentum=0.999)
        self.q2 = QuantIdentity(
            act_quant=CommonActQuant, bit_width=a_bits, return_quant_tensor=True
        )

        self.fc3 = qnn.QuantLinear(
            self.cfg[2],
            self.cfg[3],
            False,
            weight_quant=CommonWeightQuant if w_bits is not None else None,
            weight_bit_width=w_bits,
            bias_quant=None,
        )

        self.bn3 = nn.BatchNorm1d(self.cfg[1], momentum=0.999)
        self.q3 = QuantIdentity(
            act_quant=CommonActQuant, bit_width=a_bits, return_quant_tensor=True
        )

        self.fc4 = qnn.QuantLinear(
            self.cfg[3],
            self.cfg[4],
            False,
            weight_quant=CommonWeightQuant if w_bits is not None else None,
            weight_bit_width=w_bits,
        )

        for m in self.modules():
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.q1(self.bn1(self.fc1(x)))
        x = self.q2(self.bn2(self.fc2(x)))
        x = self.q3(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def prune(self, sparsity, enable):
        if not self.a_bits or not self.w_bits:
            return

        n_max_mac = 14  # int(np.floor((2**7 - 1) / (2**self.w_bits - 1) / (2**self.a_bits - 1)))
        assert n_max_mac > 0

        n_max_mac = int(n_max_mac * sparsity)

        if enable:
            prune.l1_unstructured(
                self.fc1, "weight", amount=(self.cfg[0] - n_max_mac) * self.cfg[1]
            )
            prune.l1_unstructured(
                self.fc2, "weight", amount=(self.cfg[1] - n_max_mac) * self.cfg[2]
            )
            prune.l1_unstructured(
                self.fc3, "weight", amount=(self.cfg[2] - n_max_mac) * self.cfg[3]
            )
            prune.l1_unstructured(
                self.fc4, "weight", amount=(self.cfg[3] - n_max_mac) * self.cfg[4]
            )
        else:
            prune.remove(self.fc1, "weight")
            prune.remove(self.fc2, "weight")
            prune.remove(self.fc3, "weight")
            prune.remove(self.fc4, "weight")


class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0
    signed = True


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0
