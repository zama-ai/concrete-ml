"""Quantized versions of the ONNX operators for post training quantization."""

from abc import ABC
from copy import deepcopy
from inspect import Parameter, _empty, signature
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy

from ..common.debugging import assert_true
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL
from .quantized_array import QuantizedArray

ALL_QUANTIZED_OPS: Set[Type] = set()

ONNX_OPS_TO_QUANTIZED_IMPL: Dict[str, Type["QuantizedOp"]] = {}


class QuantizedOp(ABC):
    """Base class for quantized ONNX ops implemented in numpy.

    Args:
        n_bits (int): The number of bits to use for quantization.
    """

    # impl is not optional but mypy has a long standing bug and is not able to understand this
    # properly. See https://github.com/python/mypy/issues/708#issuecomment-605636623
    impl: Optional[Callable[..., Tuple[numpy.ndarray, ...]]] = None
    n_bits: int
    output_scale: Optional[float]
    output_zero_point: Optional[int]
    constant_inputs: Dict[int, Any]
    attrs: Dict[str, Any] = {}
    _authorized_attr_names: Set[str] = set()
    # This can be used for custom implementations of some missing or (god forbid) buggy operators.
    _impl_for_op_named: Optional[str] = None
    _default_attrs: Dict[str, Any] = {}
    _params_name_to_input_idx: Dict[str, int] = {}
    _input_idx_to_params_name: Dict[int, str] = {}
    _params_that_are_onnx_inputs: Set[str] = set()
    _params_that_are_required_onnx_inputs: Set[str] = set()
    _has_attr: bool

    POSITIONAL_ARGUMENTS_KINDS = {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}

    def __init__(
        self,
        n_bits: int,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        **attrs,
    ) -> None:
        self.n_bits = n_bits
        self.output_scale = None
        self.output_zero_point = None

        constant_inputs_per_name: Dict[str, Any] = {}

        def _convert_input_name_or_idx_to_input_name(input_name_or_idx: Union[str, int]) -> str:
            if isinstance(input_name_or_idx, str):
                return input_name_or_idx
            return self._input_idx_to_params_name[input_name_or_idx]

        if constant_inputs is not None:
            # Convert input idx to op input names if needed
            constant_inputs_per_name.update(
                (
                    _convert_input_name_or_idx_to_input_name(input_name_or_idx),
                    constant_value,
                )
                for input_name_or_idx, constant_value in constant_inputs.items()
            )
            # Ignore type here as mypy has a hard time understanding what's happening.

        assert_true(
            len(
                invalid_input_names := (
                    constant_inputs_per_name.keys() - self._params_that_are_onnx_inputs
                )
            )
            == 0,
            "Got the current invalid constant input names or indices: "
            f"{', '.join(sorted(invalid_input_names))}.\n"
            f"Valid input names: {(', '.join(sorted(self._params_that_are_onnx_inputs)))}.",
        )

        # Convert input names to input indices
        self.constant_inputs = {
            self._params_name_to_input_idx[input_name]: constant_value
            for input_name, constant_value in constant_inputs_per_name.items()
        }

        assert_true(
            len(unknown_attrs := (attrs.keys() - self._authorized_attr_names)) == 0,
            f"Got the following unknown attributes: {', '.join(sorted(unknown_attrs))}. "
            + (
                f"Accepted attributes: {', '.join(sorted(self._authorized_attr_names))}."
                if len(self._authorized_attr_names) > 0
                else f"{self.__class__.__name__} does not accept attributes."
            ),
        )

        self.attrs = dict(self._default_attrs, **deepcopy(attrs))

    # Register node to our internal categories
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ALL_QUANTIZED_OPS.add(cls)
        cls._has_attr = len(cls._authorized_attr_names) > 0

        if (op_name := cls._impl_for_op_named) is not None:
            ONNX_OPS_TO_QUANTIZED_IMPL[op_name] = cls
            candidate_impl = ONNX_OPS_TO_NUMPY_IMPL.get(op_name, None)
            cls.impl = candidate_impl if candidate_impl is not None else cls.impl

        assert_true(cls.impl is not None, f"Missing 'impl' for class {cls.__name__}")

        cls._populate_op_input_infos()

    def __call__(self, *q_inputs: QuantizedArray) -> QuantizedArray:
        """Process the forward pass of the quantized op according to the implementation.

        The calibrate method needs to be called with sample data before using this function.

        Args:
            *q_inputs (QuantizedArray): Quantized inputs.

        Returns:
            QuantizedArray: Quantized output.
        """

        return self.q_impl(*q_inputs, **self.attrs)

    @classmethod
    def _populate_op_input_infos(cls):
        # for mypy
        assert cls.impl is not None
        impl_signature = signature(cls.impl)
        impl_params = impl_signature.parameters
        cls._params_name_to_input_idx = dict(reversed(val) for val in enumerate(impl_params))
        cls._input_idx_to_params_name = dict(enumerate(impl_params))
        cls._params_that_are_onnx_inputs = {
            param.name
            for param in impl_params.values()
            if param.kind in cls.POSITIONAL_ARGUMENTS_KINDS
        }
        cls._params_that_are_required_onnx_inputs = {
            param.name for param in impl_params.values() if isinstance(param.default, _empty)
        }

        cls._default_attrs = {
            param.name: param.default
            for param in impl_params.values()
            if param.kind == Parameter.KEYWORD_ONLY
        }
        cls._authorized_attr_names = set(cls._default_attrs.keys())

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Execute the quantized forward.

        Args:
            *q_inputs (QuantizedArray): Quantized inputs.
            **attrs: the QuantizedOp attributes.

        Returns:
            QuantizedArray: The returned quantized value.
        """
        f_inputs = (q_input.dequant() for q_input in q_inputs)
        prepared_inputs = self._prepare_inputs_with_constants(*f_inputs)
        f_outputs = self.call_impl(*prepared_inputs, **attrs)

        return self.quant_output(f_outputs)

    def _prepare_inputs_with_constants(self, *inputs) -> List:
        num_onnx_inputs = len(self._params_that_are_onnx_inputs)
        num_required_onnx_inputs = len(self._params_that_are_required_onnx_inputs)
        num_provided_constants = len(self.constant_inputs)
        prepared_inputs = [None] * num_onnx_inputs

        for input_idx, constant_val in self.constant_inputs.items():
            prepared_inputs[input_idx] = constant_val

        assert_true(
            num_onnx_inputs
            >= (num_inputs := len(inputs)) + num_provided_constants
            >= num_required_onnx_inputs,
            f"This operator has {num_onnx_inputs} ONNX inputs, and {num_provided_constants} "
            "constants were already provided when instantiating the class. "
            f"Got a call with {num_inputs} inputs and constants while the call expects between "
            f"{num_required_onnx_inputs} and {num_onnx_inputs} inputs and constants.",
        )

        curr_input_fill_idx = 0
        for input_ in inputs:
            while prepared_inputs[curr_input_fill_idx] is not None:
                curr_input_fill_idx += 1
            prepared_inputs[curr_input_fill_idx] = input_
            curr_input_fill_idx += 1

        return prepared_inputs

    def calibrate(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            *inputs (numpy.ndarray): Calibration sample inputs.

        Returns:
            numpy.ndarray: the output values for the provided calibration samples.
        """

        prepared_inputs = self._prepare_inputs_with_constants(*inputs)
        quantized_samples = QuantizedArray(
            self.n_bits, self.call_impl(*prepared_inputs, **self.attrs)
        )
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

    def call_impl(self, *inputs: numpy.ndarray, **attrs) -> numpy.ndarray:
        """Call self.impl to centralize mypy bug workaround.

        Args:
            *inputs (numpy.ndarray): real valued inputs.
            **attrs: the QuantizedOp attributes.

        Returns:
            numpy.ndarray: return value of self.impl
        """

        # Continuation of mypy bug
        assert self.impl is not None
        # mypy is not happy with __func__
        # self.impl will call impl with self as first parameter, so get the underlying __func__
        impl_func = self.impl.__func__  # type: ignore
        outputs = impl_func(*inputs) if not self._has_attr else impl_func(*inputs, **attrs)
        assert_true(isinstance(outputs, tuple))
        assert_true(
            (num_outputs := len(outputs)) == 1,
            f"Currently only single output ops are supported, got {num_outputs} outputs.",
        )

        return outputs[0]


class QuantizedSigmoid(QuantizedOp):
    """Quantized sigmoid op."""

    _impl_for_op_named: str = "Sigmoid"


class QuantizedRelu(QuantizedOp):
    """Quantized Relu op."""

    _impl_for_op_named: str = "Relu"


class QuantizedClip(QuantizedOp):
    """Quantized clip op."""

    _impl_for_op_named: str = "Clip"


# TODO: https://github.com/zama-ai/concrete-ml-internal/issues/195
class QuantizedGemm(QuantizedOp):
    """Quantized Gemm op."""

    _impl_for_op_named: str = "Gemm"

    def __init__(
        self,
        n_bits: int,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits, constant_inputs, **attrs)

        alpha = self.attrs.get("alpha", 1)
        beta = self.attrs.get("beta", 1)

        assert_true(
            (alpha, beta) == (1, 1),
            f"{self.__class__.__name__} currently only supports alpha == 1 and beta == 1.\n"
            f"Got alpha == {alpha} and beta == {beta}.",
        )

        assert_true(
            1 in self.constant_inputs,
            f"{self.__class__.__name__} currently only supports quantizing Gemm if weights are "
            "provided as the 'b' constant input.",
        )

    def calibrate(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of QuantizedLinear.

        Args:
            *inputs (numpy.ndarray): Inputs.

        Returns:
            numpy.ndarray: the dequantized output of the operator.
        """

        prepared_inputs = self._prepare_inputs_with_constants(*inputs)
        # Here we need the actual values of the weights and biases
        for i in range(1, len(prepared_inputs)):
            if isinstance(prepared_inputs[i], QuantizedArray):
                prepared_inputs[i] = prepared_inputs[i].values

        quantized_samples = QuantizedArray(
            self.n_bits, self.call_impl(*prepared_inputs, **self.attrs)
        )
        self.output_scale = quantized_samples.scale
        self.output_zero_point = quantized_samples.zero_point

        return quantized_samples.values

    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **_attrs,
    ) -> QuantizedArray:
        prepared_inputs = self._prepare_inputs_with_constants(*q_inputs)
        q_input: QuantizedArray = prepared_inputs[0]
        q_weights: QuantizedArray = prepared_inputs[1]
        q_bias: Optional[QuantizedArray] = prepared_inputs[2] if len(prepared_inputs) == 3 else None

        # For mypy
        assert self.output_scale is not None
        assert self.output_zero_point is not None

        # The following MatMul is done with integers, and thus, does not use of any PBS.
        # Only the final conversion to float is done with a PBS, which can actually
        # be merged with the PBS of following activation.
        # State of the art quantization method assumes the following results in a int32 accumulator.

        # Here we follow Eq.7 in https://arxiv.org/abs/1712.05877 to split the core computation
        # from the zero points and scales.

        p = q_weights.qvalues.shape[0]

        # Core matmul operation in full intergers with a shape change (INTEGERS)
        matmul = q_input.qvalues @ q_weights.qvalues

        # Sum operation in full integers resulting in large integers (INTEGERS)
        # [WORKAROUND #995] numpy.sum can't be currently done in our framework
        # sum_input = q_weights.zero_point * numpy.sum(q_input.qvalues, axis=1, keepdims=True)
        # Hack because we can't do numpy.sum(axis...,keepdims...)
        n_features = 1 if len(q_input.qvalues.shape) <= 1 else q_input.qvalues.shape[1]
        const_ones = numpy.ones(shape=(n_features, 1), dtype=numpy.int64)
        sum_input = q_weights.zero_point * (q_input.qvalues @ const_ones)

        # Last part that has to be done in FHE the rest must go in a PBS.
        # Forced fusing using .astype(numpy.float32)
        numpy_q_out = (matmul + (numpy.negative(sum_input))).astype(numpy.float32)

        # sum_weights is a constant
        sum_weights = q_input.zero_point * numpy.sum(q_weights.qvalues, axis=0, keepdims=True)

        # Quantization scales and zero points (FLOATS involved)
        # This is going to be compiled with a PBS (along with the following activation function)
        m_matmul = (q_input.scale * q_weights.scale) / (self.output_scale)

        final_term = p * q_input.zero_point * q_weights.zero_point

        numpy_q_out = numpy_q_out + final_term + (numpy.negative(sum_weights))
        numpy_q_out = m_matmul * numpy_q_out
        numpy_q_out = self.output_zero_point + numpy_q_out

        if q_bias is not None:
            bias_part = q_bias.scale / self.output_scale * (q_bias.qvalues - q_bias.zero_point)
            numpy_q_out = numpy_q_out + bias_part

        numpy_q_out = numpy.rint(numpy_q_out).clip(0, 2 ** self.n_bits - 1).astype(numpy.int64)

        return QuantizedArray(
            self.n_bits,
            numpy_q_out,
            value_is_float=False,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
        )


class QuantizedTanh(QuantizedOp):
    """Quantized Tanh op."""

    _impl_for_op_named: str = "Tanh"
