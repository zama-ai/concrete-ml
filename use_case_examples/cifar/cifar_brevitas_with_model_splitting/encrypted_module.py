import torch
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas_utils import CommonActQuant, CommonWeightQuant, TensorNorm
from constants import (
    CNV_OUT_CH_POOL,
    INTERMEDIATE_FC_FEATURES,
    KERNEL_SIZE,
    LAST_FC_IN_FEATURES,
    SPLIT_INDEX,
)
from torch.nn import AvgPool2d, BatchNorm1d, BatchNorm2d


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
