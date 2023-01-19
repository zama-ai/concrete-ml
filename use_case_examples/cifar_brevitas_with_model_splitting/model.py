# https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/models/CNV.py
import torch
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas_utils import CommonActQuant, CommonWeightQuant, TensorNorm
from torch.nn import AvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d, ModuleList

CNV_OUT_CH_POOL = [
    (64, False),
    (64, True),
    (128, False),
    (128, True),
    (256, False),
    (256, False),
]

INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3
EPSILON_VALUE = 0.5
SPLIT_INDEX = 1


# First layers of the model
# They will in-fine run in clear on the client side
# It can use float or quantization
class ClearModule(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_bit_width: int,
    ):
        super().__init__()

        self.in_ch = in_ch
        out_ch = in_ch

        self.conv_features = torch.nn.ModuleList()
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[:SPLIT_INDEX]:
            self.conv_features.append(
                Conv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=True,
                )
            )
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            if is_pool_enabled:
                self.conv_features.append(AvgPool2d(kernel_size=2))
        self.out_ch = out_ch

        self.conv_features.append(
            QuantIdentity(
                act_quant=CommonActQuant,
                return_quant_tensor=False,
                bit_width=out_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=True,
            )
        )

    def forward(self, x):
        for mod in self.conv_features:
            x = mod(x)
        return x


# Rest of the model
# They will in-fine run in FHE on the server side
# It should use quantization aware training
class EncryptedModule(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        weight_bit_width: int,
        act_bit_width: int,
        in_bit_width: int,
        in_ch: int,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.conv_features = torch.nn.ModuleList()
        self.linear_features = torch.nn.ModuleList()

        self.conv_features.append(
            QuantIdentity(  # for Q1.7 input format
                act_quant=CommonActQuant,
                return_quant_tensor=True,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=True,
                # restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        )

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[SPLIT_INDEX:]:
            self.conv_features.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            if is_pool_enabled:
                self.conv_features.append(AvgPool2d(kernel_size=2))
                self.conv_features.append(
                    QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
                )

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )

        self.linear_features.append(
            QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width,
            )
        )
        self.linear_features.append(TensorNorm())

    def forward(self, x):
        for mod in self.conv_features:
            x = mod(x)
        x = torch.flatten(x, 1)
        for mod in self.linear_features:
            x = mod(x)
        return x


# The model combining both modules
class CNV(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        weight_bit_width: int,
        act_bit_width: int,
        in_bit_width: int,
        in_ch: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.in_bit_width = in_bit_width
        self.in_ch = in_ch

        self.clear_module = ClearModule(in_ch=in_ch, out_bit_width=in_bit_width)
        self.encrypted_module = EncryptedModule(
            num_classes=num_classes,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            in_bit_width=in_bit_width,
            in_ch=self.clear_module.out_ch,
        )

    def forward(self, x):
        x = self.clear_module(x)
        return self.encrypted_module(x)
