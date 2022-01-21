"""Quantized versions of the ONNX operators for post training quantization."""

from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union

import numpy

from ..common.debugging import assert_true
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL
from .quantized_array import QuantizedArray

ALL_QUANTIZED_OPS: Set[Type] = set()
OPS_W_ATTRIBUTES: Set[Type] = set()
OPS_WO_ATTRIBUTES: Set[Type] = set()


class QuantizedOp(ABC):
    """Base class for quantized ONNX ops implemented in numpy.

    Args:
        n_bits (int): The number of bits to use for quantization.
    """

    # impl is not optional but mypy has a long standing bug and is not able to understand this
    # properly. See https://github.com/python/mypy/issues/708#issuecomment-605636623
    impl: Optional[Callable[..., Tuple[numpy.ndarray, ...]]]
    n_bits: int
    output_scale: Optional[float]
    output_zero_point: Optional[int]
    attrs: Dict[str, Any]
    _authorized_attr_names: Set[str] = set()

    def __init__(
        self,
        onnx_op_name_or_impl: Union[str, Callable[..., Tuple[numpy.ndarray, ...]]],
        n_bits: int,
        **attrs,
    ) -> None:
        self.impl = (
            ONNX_OPS_TO_NUMPY_IMPL[onnx_op_name_or_impl]
            if isinstance(onnx_op_name_or_impl, str)
            else onnx_op_name_or_impl
        )
        self.n_bits = n_bits
        self.output_scale = None
        self.output_zero_point = None

        assert_true(
            len(unknown_attrs := (attrs.keys() - self._authorized_attr_names)) == 0,
            f"Got the following unknown attributes: {', '.join(sorted(unknown_attrs))}. "
            + (
                f"Accepted attributes: {', '.join(sorted(self._authorized_attr_names))}."
                if len(self._authorized_attr_names) > 0
                else f"{self.__class__.__name__} does not accept attributes."
            ),
        )

        self.attrs = deepcopy(attrs)

    # Register node to our internal categories
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ALL_QUANTIZED_OPS.add(cls)
        if len(cls._authorized_attr_names) > 0:
            OPS_W_ATTRIBUTES.add(cls)
        else:
            OPS_WO_ATTRIBUTES.add(cls)

    def __call__(self, *q_inputs: QuantizedArray) -> QuantizedArray:
        """Process the forward pass of the quantized op according to the implementation.

        The calibrate method needs to be called with sample data before using this function.

        Args:
            *q_inputs (QuantizedArray): Quantized inputs.

        Returns:
            QuantizedArray: Quantized output.
        """

        f_inputs = (q_input.dequant() for q_input in q_inputs)
        f_outputs = self.call_impl(*f_inputs)

        return self.quant_output(f_outputs)

    def calibrate(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            *inputs (numpy.ndarray): Calibration sample inputs.

        Returns:
            numpy.ndarray: the output values for the provided calibration samples.
        """
        quantized_samples = QuantizedArray(self.n_bits, self.call_impl(*inputs))
        self.output_scale = quantized_samples.scale
        self.output_zero_point = quantized_samples.zero_point

        return quantized_samples.values

    # TODO: manage multiple inputs if it becomes necessary
    def quant_output(self, qoutput_activation: numpy.ndarray) -> QuantizedArray:
        """Quantize the output of the activation function.

        The calibrate method needs to be called with sample data before using this function.

        Args:
            qoutput_activation (numpy.ndarray): Output of the activation function.

        Returns:
            QuantizedArray: Quantized output.
        """

        assert_true(
            self.output_scale is not None,
            f"output_scale was None for class {self.__class__.__name__}, "
            "did you forget to call calibrate with sample data?",
        )
        assert_true(
            self.output_zero_point is not None,
            f"output_zero_point was None for class {self.__class__.__name__}, "
            "did you forget to call calibrate with sample data?",
        )

        # for mypy
        assert self.output_scale is not None and self.output_zero_point is not None

        qoutput_activation = qoutput_activation / self.output_scale + self.output_zero_point
        qoutput_activation = (
            numpy.rint(qoutput_activation).clip(0, 2 ** self.n_bits - 1).astype(numpy.int64)
        )

        return QuantizedArray(
            self.n_bits,
            qoutput_activation,
            value_is_float=False,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
        )

    def call_impl(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Call self.impl to centralize mypy bug workaround.

        Args:
            *inputs (numpy.ndarray): real valued inputs.

        Returns:
            numpy.ndarray: return value of self.impl
        """
        # Continuation of mypy bug
        assert self.impl is not None
        outputs = self.impl(*inputs, **self.attrs)
        assert_true(isinstance(outputs, tuple))
        assert_true(
            (num_outputs := len(outputs)) == 1,
            f"Currently only single output ops are supported, got {num_outputs} outputs.",
        )

        return outputs[0]


class QuantizedSigmoid(QuantizedOp):
    """Quantized sigmoid op."""

    def __init__(self, n_bits: int) -> None:
        super().__init__("Sigmoid", n_bits)


class QuantizedRelu(QuantizedOp):
    """Quantized Relu op."""

    def __init__(self, n_bits: int) -> None:
        super().__init__("Relu", n_bits)


class QuantizedClip(QuantizedOp):
    """Quantized clip op."""

    _authorized_attr_names: Set[str] = set(("min", "max"))

    def __init__(self, n_bits: int, **attrs) -> None:
        super().__init__("Clip", n_bits, **attrs)


assert_true(
    ALL_QUANTIZED_OPS == OPS_W_ATTRIBUTES.union(OPS_WO_ATTRIBUTES),
    "Error while checking ops divided according to having attributes",
)
