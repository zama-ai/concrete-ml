"""Quantized versions of the ONNX operators for post training quantization."""

from abc import ABC
from copy import deepcopy
from inspect import Parameter, _empty, signature
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import numpy
from concrete.common.extensions import convolution as hconv

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

        if (op_name := cls._impl_for_op_named) is not None:
            ONNX_OPS_TO_QUANTIZED_IMPL[op_name] = cls
            candidate_impl = ONNX_OPS_TO_NUMPY_IMPL.get(op_name, None)
            cls.impl = candidate_impl if candidate_impl is not None else cls.impl

        assert_true(cls.impl is not None, f"Missing 'impl' for class {cls.__name__}")

        cls._populate_op_input_infos()
        cls._has_attr = len(cls._authorized_attr_names) > 0

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
        # Here we need the float32 values from the QuantizedArrays. By default, when possible,
        # we want QuantizedOps to convert to TLUs. Ops that need to do quantized computation
        # will call _prepare_inputs_with_constants with quantize_actual_values=True
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=False
        )
        f_outputs = self.call_impl(*prepared_inputs, **attrs)
        return self.prepare_output(f_outputs)

    def _prepare_inputs_with_constants(
        self,
        *inputs: Union[QuantizedArray, numpy.ndarray],
        calibrate: bool,
        quantize_actual_values: bool,
    ) -> List:
        """Retrieve all the inputs of an operator in the computational graph.

        This helper method will prepare a list of inputs to an operator. Inputs can be variables,
        i.e. encrypted tensors, or constants (in the clear). Inputs to an operator are set-up in
        the slots of a list, as the order of inputs is important.

        Usually the input list is populated with QuantizedArrays. Operators that require the
        original float (operators that only produce or contribute to TLUs) values will just read
        out the .values of  these quantized arrays. Operators that do matrix multiplication will
        read out the quantized integer values of the arrays.  The method can be called during
        calibration, in which case the variable inputs are just float numpy tensors.

        Args:
             *inputs (Union[QuantizedArray, numpy.ndarray]): A list of all variable inputs
            calibrate (bool): A flag specifying if the method is called during calibration
            quantize_actual_values (bool): If called by a quantized operator that does matrix
                multiplication between encrypted and clear values, this method will apply
                the quantization computation to the input, which will be fused in a (potentially
                larger) TLU, with preceding floating point computations

        Returns:
            result (List): a list of inputs which are either QuantizedArray or numpy.arrays. If
                quantize_actual_values==True then the quantization code is applied
        """
        num_onnx_inputs = len(self._params_that_are_onnx_inputs)
        num_required_onnx_inputs = len(self._params_that_are_required_onnx_inputs)
        num_provided_constants = len(self.constant_inputs)
        prepared_inputs: List[Optional[Union[QuantizedArray, numpy.ndarray]]] = [
            None
        ] * num_onnx_inputs

        # If calibrating (calibrate=True): inputs are numpy.ndarrays of float32
        # If used in the computation graph (calibrate=False): inputs are QuantizedArrays, of which
        # we use the float32 .values and optionally we quantized them to int

        # Constants are quantized during graph creation. If calibrating, pass through
        # the original float values. If running in the computation graph, if the quantized values
        # are requested, use the QuantizedArray object. Finally, if running the graph and
        # the original float values are requested, then return the QuantizedArray.values
        for input_idx, constant_val in self.constant_inputs.items():
            prepared_inputs[input_idx] = (
                constant_val if not calibrate and quantize_actual_values else constant_val.values
            )

        assert_true(
            num_onnx_inputs
            >= (num_inputs := len(inputs)) + num_provided_constants
            >= num_required_onnx_inputs,
            f"This operator has {num_onnx_inputs} ONNX inputs, and {num_provided_constants} "
            "constants were already provided when instantiating the class. "
            f"Got a call with {num_inputs} inputs and constants while the call expects between "
            f"{num_required_onnx_inputs} and {num_onnx_inputs} inputs and constants.",
        )

        # If calibrating, the function is called with numpy.ndarrays
        # If not calibrating, the function is called with QuantizedArray inputs
        # If quantized values are requested, we quantized the float32 values contained in the
        # QuantizedArrays, else we return the float32 values directly.

        curr_input_fill_idx = 0
        for input_ in inputs:
            while prepared_inputs[curr_input_fill_idx] is not None:
                curr_input_fill_idx += 1

            if calibrate:
                prepared_inputs[curr_input_fill_idx] = input_
            elif quantize_actual_values:
                # Here we want to trace the code that does quantization, to produce a TLU
                input_ = cast(QuantizedArray, input_)
                input_.quant()
                prepared_inputs[curr_input_fill_idx] = input_
            else:
                input_ = cast(QuantizedArray, input_)
                prepared_inputs[curr_input_fill_idx] = input_.values

            curr_input_fill_idx += 1

        return prepared_inputs

    def calibrate(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            *inputs (numpy.ndarray): Calibration sample inputs.

        Returns:
            numpy.ndarray: the output values for the provided calibration samples.
        """

        # Here we need the actual values of the constants, we need to pass through
        # the numpy.ndarrays in the computation graph
        prepared_inputs = self._prepare_inputs_with_constants(
            *inputs, calibrate=True, quantize_actual_values=False
        )
        quantized_samples = QuantizedArray(
            self.n_bits, self.call_impl(*prepared_inputs, **self.attrs)
        )
        self.output_scale = quantized_samples.scale
        self.output_zero_point = quantized_samples.zero_point

        return quantized_samples.values

    # TODO: manage multiple inputs if it becomes necessary
    def prepare_output(self, qoutput_activation: numpy.ndarray) -> QuantizedArray:
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
        return QuantizedArray(
            self.n_bits,
            qoutput_activation,
            value_is_float=True,
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
        assert_true(
            isinstance(outputs, tuple),
            f"The output of {impl_func.__name__} needs to be a tuple. Got {outputs}",
        )
        assert_true(
            (num_outputs := len(outputs)) == 1,
            f"Currently only single output ops are supported, got {num_outputs} outputs.",
        )

        return outputs[0]


class QuantizedSigmoid(QuantizedOp):
    """Quantized sigmoid op."""

    _impl_for_op_named: str = "Sigmoid"


class QuantizedHardSigmoid(QuantizedOp):
    """Quantized HardSigmoid op."""

    _impl_for_op_named: str = "HardSigmoid"


class QuantizedRelu(QuantizedOp):
    """Quantized Relu op."""

    _impl_for_op_named: str = "Relu"


class QuantizedLeakyRelu(QuantizedOp):
    """Quantized LeakyRelu op."""

    _impl_for_op_named: str = "LeakyRelu"


class QuantizedElu(QuantizedOp):
    """Quantized Elu op."""

    _impl_for_op_named: str = "Elu"


class QuantizedSelu(QuantizedOp):
    """Quantized Selu op."""

    _impl_for_op_named: str = "Selu"


class QuantizedCelu(QuantizedOp):
    """Quantized Celu op."""

    _impl_for_op_named: str = "Celu"


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
            alpha == 1 and beta in [0, 1],
            f"{self.__class__.__name__} currently only supports alpha == 1 and beta in [0, 1].\n"
            f"Got alpha == {alpha} and beta == {beta}.",
        )

        assert_true(
            1 in self.constant_inputs,
            f"{self.__class__.__name__} currently only supports quantizing "
            f"{self._impl_for_op_named} if weights are provided as the 'b' constant input.",
        )

    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **attrs,
    ) -> QuantizedArray:

        alpha = self.attrs.get("alpha", 1)
        beta = self.attrs.get("beta", 1)

        # If alpha != 1 or beta not in [0, 1], this function must be modified
        assert_true(alpha == 1)
        assert_true(beta in [0, 1])

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input: QuantizedArray = prepared_inputs[0]
        q_weights: QuantizedArray = prepared_inputs[1]
        q_bias: Optional[QuantizedArray] = (
            None if len(prepared_inputs) == 2 or beta == 0 else prepared_inputs[2]
        )

        # Using snake case here to please the python format, the original attrs don't have the '_'
        # Use default false so we also support MatMul impl, MatMul does not have these flags
        transpose_inputs = attrs.get("transA", False)
        transpose_w = attrs.get("transB", False)

        input_q_values = numpy.transpose(q_input.qvalues) if transpose_inputs else q_input.qvalues
        weights_q_values = numpy.transpose(q_weights.qvalues) if transpose_w else q_weights.qvalues

        # For mypy
        assert self.output_scale is not None
        assert self.output_zero_point is not None

        # The following MatMul is done with integers, and thus, does not use of any PBS.
        # Rescaling the output of the integer MatMul to handle scale changes is done
        # in float32 and will thus be fused with any float32 processing that follows this layer.

        # Here we follow Eq.7 in https://arxiv.org/abs/1712.05877 to split the core computation
        # from the zero points and scales.

        p = weights_q_values.shape[0]

        # Core matmul operation in full intergers with a shape change (INTEGERS)
        matmul = input_q_values @ weights_q_values

        # Sum operation in full integers resulting in large integers (INTEGERS)
        # [WORKAROUND #995] numpy.sum can't be currently done in our framework
        # sum_input = q_weights.zero_point * numpy.sum(input_q_values, axis=1, keepdims=True)
        # Hack because we can't do numpy.sum(axis...,keepdims...)
        n_features = 1 if len(input_q_values.shape) <= 1 else input_q_values.shape[1]
        const_ones = numpy.ones(shape=(n_features, 1), dtype=numpy.int64)
        sum_input = -q_weights.zero_point * (input_q_values @ const_ones)

        # Last part that has to be done in integer, the rest must go in a PBS.
        # Forced fusing using .astype(numpy.float32)
        numpy_q_out = (matmul + sum_input).astype(numpy.float32)

        # sum_weights is a constant
        sum_weights = q_input.zero_point * numpy.sum(weights_q_values, axis=0, keepdims=True)

        final_term = p * q_input.zero_point * q_weights.zero_point

        numpy_q_out = numpy_q_out + final_term + (numpy.negative(sum_weights))

        # Quantization scales and zero points (FLOATS involved)
        # This is going to be compiled with a PBS (along with the following activation function)

        # Note that here we do not rescale to the output_scale and we do not add a zero-point
        # Any following Gemm/MatMul/Conv layers will do the rescaling (during requantization)
        # by calling _prepare_inputs_with_constants(...quantize_real_values=True)
        m_matmul = q_input.scale * q_weights.scale
        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None:
            # The bias is handled as a float32 and will be fused
            numpy_q_out = numpy_q_out + q_bias.values

        # Return the float32 values, so that CN can fuse any following float32 operations
        # We also keep track of the scaling factor and zero-point, since these will be
        # applied by the following layers.
        return QuantizedArray(
            self.n_bits,
            numpy_q_out,
            value_is_float=True,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
        )


class QuantizedMatMul(QuantizedGemm):
    """Quantized MatMul op."""

    _impl_for_op_named: str = "MatMul"


class QuantizedAdd(QuantizedOp):
    """Quantized Addition operator.

    Can add either two variables (both encrypted) or a variable and a constant
    """

    _impl_for_op_named: str = "Add"

    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **attrs,
    ) -> QuantizedArray:

        # For mypy
        assert self.output_scale is not None
        assert self.output_zero_point is not None

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input_0: QuantizedArray = prepared_inputs[0]
        q_input_1: QuantizedArray = prepared_inputs[1]

        # Optimize computation when adding constants, using a TLU and no quantization
        # Both inputs can not be constant since constant folding would eliminate the Add node
        if len(self.constant_inputs) > 0:
            assert_true(len(self.constant_inputs) == 1)
            const_idx = list(self.constant_inputs.keys())[0]
            encrypted_idx = 1 - const_idx

            return QuantizedArray(
                self.n_bits,
                prepared_inputs[0].values + prepared_inputs[1].values,
                is_signed=prepared_inputs[encrypted_idx].is_signed,
                value_is_float=True,
                scale=self.output_scale,
                zero_point=self.output_zero_point,
            )

        # De-quantize with input params and re-quantize with output parameters
        # This will use TLUs over each element of the two inputs
        # We do the dequantization directly, instead of q_inputs[0].dequant(),
        # So that we do not lose precision in the computation

        rescale_q0 = numpy.rint(
            q_input_0.scale / self.output_scale * (q_input_0.qvalues + (-q_input_0.zero_point))
            + self.output_zero_point
        ).astype(numpy.int64)

        rescale_q1 = numpy.rint(
            q_input_1.scale / self.output_scale * (q_input_1.qvalues + (-q_input_1.zero_point))
            + self.output_zero_point
        ).astype(numpy.int64)

        # The sum of quantized encrypted integer values
        # This sum has << max(in_bits0, in_bits1) + 1 >> bits
        # Moreover, the zeropoint will be sum of input zeropoints
        sum_q = rescale_q0 + rescale_q1

        # But we would like the output to have n_bits, so we de-quantize
        dequant_sum = self.output_scale * (sum_q + (-2 * self.output_zero_point))

        # Return the raw float32 values without requantizing them to the new scale, as any
        # following Gemm/Add/Conv will quantize them with _prepare_inputs_with_constants(...)
        return QuantizedArray(
            self.n_bits,
            dequant_sum,
            value_is_float=True,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
        )


class QuantizedTanh(QuantizedOp):
    """Quantized Tanh op."""

    _impl_for_op_named: str = "Tanh"


class QuantizedSoftplus(QuantizedOp):
    """Quantized Softplus op."""

    _impl_for_op_named: str = "Softplus"


class QuantizedExp(QuantizedOp):
    """Quantized Exp op."""

    _impl_for_op_named: str = "Exp"


class QuantizedLog(QuantizedOp):
    """Quantized Log op."""

    _impl_for_op_named: str = "Log"


class QuantizedAbs(QuantizedOp):
    """Quantized Abs op."""

    _impl_for_op_named: str = "Abs"


class QuantizedLinear(QuantizedGemm):
    """Helper Class to have the equivalent layer to torch.nn.Linear."""

    _impl_for_op_named: str = "Linear"

    def __init__(
        self,
        n_bits: int,
        q_weights: QuantizedArray,
        q_bias: Optional[QuantizedArray] = None,
    ) -> None:
        constant_inputs = {"b": q_weights} if q_bias is None else {"b": q_weights, "c": q_bias}
        super().__init__(n_bits, constant_inputs=constant_inputs)


class QuantizedIdentity(QuantizedOp):
    """Quantized Identity op."""

    _impl_for_op_named: str = "Identity"

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        assert_true(len(q_inputs) == 1, "Identity does not work with multiple QuantizedArray")
        self.output_scale = q_inputs[0].scale
        self.output_zero_point = q_inputs[0].zero_point
        return super().q_impl(*q_inputs, **attrs)


class QuantizedReshape(QuantizedOp):
    """Quantized Reshape op."""

    _impl_for_op_named: str = "Reshape"

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Reshape the input integer encrypted tensor.

        Args:
            q_inputs: an encrypted integer tensor at index 0 and one constant shape at index 1
            attrs: additional optional reshape options

        Returns:
            result (QuantizedArray): reshaped encrypted integer tensor
        """

        # FIXME: Currently reshape quantizes the inputs, but this is unnecessary if the reshape
        # operation could be fused into a Gemm/Add/Conv that follows it. We should reshape
        # here only if the reshaped result is returned directly from the FHE program.
        # See https://github.com/zama-ai/concrete-ml-internal/issues/527
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        newshape = prepared_inputs[1].values

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].n_bits,
            numpy.reshape(prepared_inputs[0].qvalues, newshape),
            value_is_float=False,
            scale=prepared_inputs[0].scale,
            zero_point=prepared_inputs[0].zero_point,
        )


class QuantizedConv(QuantizedOp):
    """Quantized Conv op."""

    _impl_for_op_named: str = "Conv"

    def __init__(
        self,
        n_bits: int,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        **attrs,
    ) -> None:
        """Construct the quantized convolution operator and retrieve parameters.

        Args:
            n_bits: number of bits for output quantization
            constant_inputs: the weights and activations
            attrs: convolution options
                dilations (Tuple[int]): dilation of the kernel, default 1 on all dimensions.
                group (int): number of convolution groups, default 1
                kernel_shape (Tuple[int]): shape of the kernel. Should have 2 elements for 2d conv
                pads (Tuple[int]): padding in ONNX format (begin, end) on each axis
                strides (Tuple[int]): stride of the convolution on each axis
        """

        super().__init__(n_bits, constant_inputs, **attrs)

        # Get the ONNX parameters
        self.dilations = attrs.get("dilations", (1, 1))
        self.group = attrs.get("group", 1)
        self.kernel_shape = attrs.get("kernel_shape", None)
        self.pads = attrs.get("pads", (0, 0, 0, 0))
        self.strides = attrs.get("strides", (1, 1))

        # Validate the parameters
        assert_true(
            len(self.kernel_shape) == 2, "The convolution operator currently supports only 2d"
        )
        assert_true(
            len(self.kernel_shape) == len(self.strides),
            "The convolution operator requires the number of strides to "
            "be the same as the number of kernel dimensions",
        )
        assert_true(
            bool(numpy.all(numpy.asarray(self.dilations) == 1)),
            "The convolution operator in Concrete Numpy does not suppport dilation",
        )
        assert_true(
            self.group == 1, "The convolution operator in Concrete Numpy does not support groups"
        )
        assert_true(
            len(self.pads) == 2 * len(self.kernel_shape),
            "The convolution operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard",
        )
        assert_true(
            self.pads[0] == self.pads[len(self.kernel_shape)]
            and self.pads[1] == self.pads[1 + len(self.kernel_shape)],
            "The convolution operator in Concrete ML only supports symmetric padding",
        )
        assert_true(
            self.pads[0] == 0 and self.pads[1] == 0,
            "Quantized convolution only supports 0-padding convolution for now",
        )

    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **attrs,
    ) -> QuantizedArray:
        """Compute the quantized convolution between two quantized tensors.

        Allows an optional quantized bias.

        Args:
            q_inputs: input tuple, contains
                x (numpy.ndarray): input data. Shape is N x C x H x W for 2d
                w (numpy.ndarray): weights tensor. Shape is (O x I x Kh x Kw) for 2d
                b (numpy.ndarray, Optional): bias tensor, Shape is (O,)
            attrs: convolution options handled in constructor

        Returns:
            res (QuantizedArray): result of the quantized integer convolution
        """

        # For mypy
        assert self.output_scale is not None
        assert self.output_zero_point is not None

        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]
        q_weights: QuantizedArray = prepared_inputs[1]
        q_bias: Optional[QuantizedArray] = None if len(prepared_inputs) == 2 else prepared_inputs[2]

        # Prepare a constant tensor to compute the sum of the inputs
        q_weights_1 = numpy.ones_like(q_weights.qvalues)

        # We follow the Quantized Gemm implementation
        # which in turn follows Eq.7 in https://arxiv.org/abs/1712.05877
        # to split the core computation from the zero points and scales.

        # Compute the first encrypted term that convolves weights and inputs
        conv_wx = hconv.conv2d(
            q_input.qvalues,
            q_weights.qvalues,
            None,
            self.pads,
            self.strides,
            self.dilations,
        )

        # Compute the sum of the inputs (second encrypted term)
        zw_conv_1x = -q_weights.zero_point * hconv.conv2d(
            q_input.qvalues,
            q_weights_1,
            None,
            self.pads,
            self.strides,
            self.dilations,
        )

        # The total number of elements that are convolved by the application of a single kernel
        n_weights = numpy.prod(q_weights.qvalues.shape[1:])

        # Last part that has to be done in FHE the rest must go in a PBS.
        # Forced fusing using .astype(numpy.float32)
        numpy_q_out = (conv_wx + zw_conv_1x).astype(numpy.float32)

        # Compute the third term, the sum of the weights which is a constant
        sum_weights = q_input.zero_point * numpy.sum(
            q_weights.qvalues, axis=(1, 2, 3), keepdims=True
        ).transpose(1, 0, 2, 3)

        # Compute the forth term which is a constant
        final_term = n_weights * q_input.zero_point * q_weights.zero_point

        # Now compute the whole sum (sum of the four terms)
        numpy_q_out = numpy_q_out + final_term + (numpy.negative(sum_weights))

        # Compute the rescaling factor that dequantizes the input
        # This is going to be compiled with a PBS (along with the following activation function)
        # Note that we don't requantize the output of the conv, this will be done by
        # any Gemm/Add/Conv layers that follow
        m_matmul = q_input.scale * q_weights.scale

        # Rescale from scale=scale_inputs x scale_outputs to output scale
        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None:
            # The bias addition is handled in float and will be fused into a TLU
            # Broadcast the rescaled biases to each channel
            numpy_q_out = numpy_q_out + q_bias.values.reshape((1, -1, 1, 1))  # bias_part

        bias_is_signed = q_bias.is_signed if q_bias is not None else False
        # And return as a QuantizedArray initialized from the float32 data, keeping
        # track of the quantization parameters
        return QuantizedArray(
            self.n_bits,
            numpy_q_out,
            is_signed=q_input.is_signed or q_weights.is_signed or bias_is_signed,
            value_is_float=True,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
        )
