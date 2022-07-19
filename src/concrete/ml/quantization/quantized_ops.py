"""Quantized versions of the ONNX operators for post training quantization."""

# pylint: disable=too-many-lines
# FIXME: #1018

from typing import Any, Dict, Optional, Sequence, Set, SupportsIndex, Union

import numpy
from concrete.onnx import conv as cnp_conv

from ..common.debugging import assert_true
from .base_quantized_op import QuantizedOp
from .quantized_array import QuantizationOptions, QuantizedArray


class QuantizedSigmoid(QuantizedOp):
    """Quantized sigmoid op."""

    _impl_for_op_named: str = "Sigmoid"


class QuantizedHardSigmoid(QuantizedOp):
    """Quantized HardSigmoid op."""

    _impl_for_op_named: str = "HardSigmoid"


class QuantizedRelu(QuantizedOp):
    """Quantized Relu op."""

    _impl_for_op_named: str = "Relu"


class QuantizedPRelu(QuantizedOp):
    """Quantized PRelu op."""

    _impl_for_op_named: str = "PRelu"


class QuantizedLeakyRelu(QuantizedOp):
    """Quantized LeakyRelu op."""

    _impl_for_op_named: str = "LeakyRelu"


class QuantizedHardSwish(QuantizedOp):
    """Quantized Hardswish op."""

    _impl_for_op_named: str = "HardSwish"


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


class QuantizedRound(QuantizedOp):
    """Quantized round op."""

    _impl_for_op_named: str = "Round"


class QuantizedPow(QuantizedOp):
    """Quantized pow op.

    Only works for a float constant power. This operation will be fused to a (potentially
    larger) TLU.
    """

    _impl_for_op_named: str = "Pow"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support power raising between encrypted tensors
        # Only power-raising between
        # - encrypted tensors and float constants
        # - tensors that are produced by a unique integer tensor
        # is supported
        # Power-raising between two constants is possible but should be optimized out by
        # the constant folding procedure
        assert_true(self.can_fuse() or (constant_inputs is not None and len(constant_inputs) == 1))

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Power raising can be fused and computed in float when a single integer tensor generates
        both the operands. For example in the formula: f(x) = x ** (x + 1)  where x is an integer
        tensor.

        Returns:
            bool: Can fuse
        """
        return len(self._int_input_names) == 1


# TODO: https://github.com/zama-ai/concrete-ml-internal/issues/195
class QuantizedGemm(QuantizedOp):
    """Quantized Gemm op."""

    _impl_for_op_named: str = "Gemm"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

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
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None

        assert q_weights.quantizer.scale is not None
        assert q_weights.quantizer.zero_point is not None

        assert q_input.quantizer.scale is not None
        assert q_input.quantizer.zero_point is not None

        # The following MatMul is done with integers, and thus, does not use of any PBS.
        # Rescaling the output of the integer MatMul to handle scale changes is done
        # in float32 and will thus be fused with any float32 processing that follows this layer.

        # Here we follow Eq.7 in https://arxiv.org/abs/1712.05877 to split the core computation
        # from the zero points and scales.

        p = weights_q_values.shape[0]

        # Core matmul operation in full intergers with a shape change (INTEGERS)
        matmul = input_q_values @ weights_q_values

        # If the weights have symmetric quantization, their zero point will be 0
        # The following check avoids the computation of the sum of the inputs, which may have
        # large bitwidth, in the case where it would be multiplied by zero
        if q_weights.quantizer.zero_point != 0:
            # Sum operation in full integers resulting in large integers (INTEGERS)
            sum_input = -q_weights.quantizer.zero_point * numpy.sum(
                input_q_values, axis=1, keepdims=True
            )

            # Last part that has to be done in integer, the rest must go in a PBS.
            # Forced fusing using .astype(numpy.float32)
            numpy_q_out = (matmul + sum_input).astype(numpy.float32)
        else:
            numpy_q_out = matmul.astype(numpy.float32)

        # sum_weights is a constant
        sum_weights = q_input.quantizer.zero_point * numpy.sum(
            weights_q_values, axis=0, keepdims=True
        )

        final_term = p * q_input.quantizer.zero_point * q_weights.quantizer.zero_point

        numpy_q_out = numpy_q_out + final_term + (numpy.negative(sum_weights))

        # Quantization scales and zero points (FLOATS involved)
        # This is going to be compiled with a PBS (along with the following activation function)

        # Note that here we do not rescale to the output_scale and we do not add a zero-point
        # Any following Gemm/MatMul/Conv layers will do the rescaling (during requantization)
        # by calling _prepare_inputs_with_constants(...quantize_real_values=True)
        m_matmul = q_input.quantizer.scale * q_weights.quantizer.scale
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
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def can_fuse(self):
        """Determine if this op can be fused.

        Gemm operation can not be fused since it must be performed over integer tensors and it
        combines different values of the input tensors.

        Returns:
            bool: False, this operation can not be fused as it adds different encrypted integers
        """

        return False


class QuantizedMatMul(QuantizedGemm):
    """Quantized MatMul op."""

    _impl_for_op_named: str = "MatMul"


class QuantizedAdd(QuantizedOp):
    """Quantized Addition operator.

    Can add either two variables (both encrypted) or a variable and a constant
    """

    _impl_for_op_named: str = "Add"
    b_sign: int = 1

    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **attrs,
    ) -> QuantizedArray:

        # For mypy
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None
        assert self.b_sign in [-1, 1]

        # Optimize computation when adding constants, or tensors obtained from a unique integer
        # tensor. Optimization allows univariate float subgraph fusion to a TLU
        execute_in_float = len(self.constant_inputs) > 0 or self.can_fuse()
        assert_true(
            len(self.constant_inputs) < 2,
            "Constant folding should have eliminated a two constant-input add node",
        )

        if execute_in_float:
            prepared_inputs = self._prepare_inputs_with_constants(
                *q_inputs, calibrate=False, quantize_actual_values=False
            )

            return QuantizedArray(
                self.n_bits,
                prepared_inputs[0] + self.b_sign * prepared_inputs[1],
                value_is_float=True,
                options=self._get_output_quant_opts(),
                stats=self.output_quant_stats,
                params=self.output_quant_params,
            )

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input_0: QuantizedArray = prepared_inputs[0]
        q_input_1: QuantizedArray = prepared_inputs[1]

        assert q_input_0.quantizer.scale is not None
        assert q_input_0.quantizer.zero_point is not None

        assert q_input_1.quantizer.scale is not None
        assert q_input_1.quantizer.zero_point is not None

        # De-quantize with input params and re-quantize with output parameters
        # This will use TLUs over each element of the two inputs
        # We do the dequantization directly, instead of q_inputs[0].dequant(),
        # So that we do not lose precision in the computation

        rescale_q0 = numpy.rint(
            q_input_0.quantizer.scale
            / self.output_quant_params.scale
            * (q_input_0.qvalues + (-q_input_0.quantizer.zero_point))
        ).astype(numpy.int64)

        rescale_q1 = numpy.rint(
            q_input_1.quantizer.scale
            / self.output_quant_params.scale
            * (q_input_1.qvalues + (-q_input_1.quantizer.zero_point))
        ).astype(numpy.int64)

        # The sum of quantized encrypted integer values
        # This sum has << max(in_bits0, in_bits1) + 1 >> bits
        # Moreover, the zeropoint will be sum of input zeropoints
        if self.b_sign == 1:
            sum_q = rescale_q0 + rescale_q1
        elif self.b_sign == -1:
            sum_q = rescale_q0 + (-1) * rescale_q1

        # But we would like the output to have n_bits, so we de-quantize
        dequant_sum = self.output_quant_params.scale * sum_q

        # Return the raw float32 values without re-quantizing them to the new scale, as any
        # following Gemm/Add/Conv will quantize them with _prepare_inputs_with_constants(...)
        return QuantizedArray(
            self.n_bits,
            dequant_sum,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Add operation can be computed in float and fused if it operates over inputs produced
        by a single integer tensor. For example the expression x + x * 1.75, where x is
        an encrypted tensor, can be computed with a single TLU.

        Returns:
            bool: Whether the number of integer input tensors allows computing this op as a TLU
        """

        return len(self._int_input_names) == 1


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


class QuantizedIdentity(QuantizedOp):
    """Quantized Identity op."""

    _impl_for_op_named: str = "Identity"

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        assert_true(len(q_inputs) == 1, "Identity does not work with multiple QuantizedArray")
        self.output_quant_params = q_inputs[0].quantizer.quant_params
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

        # FIXME: Remove this cast when #1141 is fixed.
        # (https://github.com/zama-ai/concrete-ml-internal/issues/1141)
        newshape = prepared_inputs[1].values.astype(numpy.int64)

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            numpy.reshape(prepared_inputs[0].qvalues, newshape),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )


class QuantizedConv(QuantizedOp):
    """Quantized Conv op."""

    _impl_for_op_named: str = "Conv"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        """Construct the quantized convolution operator and retrieve parameters.

        Args:
            n_bits_output: number of bits for the quantization of the outputs of this operator
            int_input_names: names of integer tensors that are taken as input for this operation
            constant_inputs: the weights and activations
            input_quant_opts: options for the input quantizer
            attrs: convolution options
                dilations (Tuple[int]): dilation of the kernel, default 1 on all dimensions.
                group (int): number of convolution groups, default 1
                kernel_shape (Tuple[int]): shape of the kernel. Should have 2 elements for 2d conv
                pads (Tuple[int]): padding in ONNX format (begin, end) on each axis
                strides (Tuple[int]): stride of the convolution on each axis
        """

        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

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
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None

        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]
        q_weights: QuantizedArray = prepared_inputs[1]
        q_bias: Optional[QuantizedArray] = None if len(prepared_inputs) == 2 else prepared_inputs[2]

        # Prepare a constant tensor to compute the sum of the inputs
        q_weights_1 = numpy.ones_like(q_weights.qvalues)

        assert q_weights.quantizer.scale is not None
        assert q_weights.quantizer.zero_point is not None

        assert q_input.quantizer.scale is not None
        assert q_input.quantizer.zero_point is not None

        # We follow the Quantized Gemm implementation
        # which in turn follows Eq.7 in https://arxiv.org/abs/1712.05877
        # to split the core computation from the zero points and scales.

        # Compute the first encrypted term that convolves weights and inputs
        conv_wx = cnp_conv(
            q_input.qvalues,
            q_weights.qvalues,
            None,
            self.pads,
            self.strides,
            self.dilations,
        )

        # The total number of elements that are convolved by the application of a single kernel
        n_weights = numpy.prod(q_weights.qvalues.shape[1:])

        # If the weights have symmetric quantization, their zero point will be 0
        # The following check avoids the computation of the sum of the inputs, which may have
        # large bitwidth, in the case where it would be multiplied by zero
        if q_weights.quantizer.zero_point != 0:
            # Compute the sum of the inputs (second encrypted term)
            zw_conv_1x = -q_weights.quantizer.zero_point * cnp_conv(
                q_input.qvalues,
                q_weights_1,
                None,
                self.pads,
                self.strides,
                self.dilations,
            )

            # Last part that has to be done in FHE the rest must go in a PBS.
            # Forced fusing using .astype(numpy.float32)
            numpy_q_out = (conv_wx + zw_conv_1x).astype(numpy.float32)
        else:
            numpy_q_out = conv_wx.astype(numpy.float32)

        # Compute the third term, the sum of the weights which is a constant
        sum_weights = q_input.quantizer.zero_point * numpy.sum(
            q_weights.qvalues, axis=(1, 2, 3), keepdims=True
        ).transpose(1, 0, 2, 3)

        # Compute the forth term which is a constant
        final_term = n_weights * q_input.quantizer.zero_point * q_weights.quantizer.zero_point

        # Now compute the whole sum (sum of the four terms)
        numpy_q_out = numpy_q_out + final_term + (numpy.negative(sum_weights))

        # Compute the rescaling factor that dequantizes the input
        # This is going to be compiled with a PBS (along with the following activation function)
        # Note that we don't requantize the output of the conv, this will be done by
        # any Gemm/Add/Conv layers that follow
        m_matmul = q_input.quantizer.scale * q_weights.quantizer.scale

        # Rescale from scale=scale_inputs x scale_outputs to output scale
        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None:
            # The bias addition is handled in float and will be fused into a TLU
            # Broadcast the rescaled biases to each channel
            numpy_q_out = numpy_q_out + q_bias.values.reshape((1, -1, 1, 1))  # bias_part

        # And return as a QuantizedArray initialized from the float32 data, keeping
        # track of the quantization parameters
        return QuantizedArray(
            self.n_bits,
            numpy_q_out,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Conv operation can not be fused since it must be performed over integer tensors and it
        combines different elements of the input tensors.

        Returns:
            bool: False, this operation can not be fused as it adds different encrypted integers
        """

        return False


class QuantizedAvgPool(QuantizedOp):
    """Quantized Average Pooling op."""

    _impl_for_op_named: str = "AveragePool"

    # Since this op takes a single input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:

        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # Get the ONNX parameters
        self.ceil_mode = attrs.get("ceil_mode", None)
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
        assert_true(
            self.ceil_mode == 0,
            "Average Pooling only supports Torch-style dimension computation with ceil_mode=0",
        )

        self.kernel: Union[numpy.ndarray, None] = None
        self.norm_const: Union[float, None] = None

    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **attrs,
    ) -> QuantizedArray:

        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]

        n_in_channels = q_input.qvalues.shape[1]
        kernel = numpy.zeros(
            (n_in_channels, n_in_channels, self.kernel_shape[0], self.kernel_shape[1]),
            dtype=numpy.uint8,
        )
        for i in range(n_in_channels):
            kernel[i, i, ::] = 1

        norm_const = 1.0 / numpy.prod(self.kernel_shape)

        sum_result = cnp_conv(q_input.qvalues, kernel, None, self.pads, self.strides)

        result = (
            sum_result.astype(numpy.float32) * norm_const - q_input.quantizer.zero_point
        ) * q_input.quantizer.scale

        return QuantizedArray(
            self.n_bits,
            result,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Avg Pooling operation can not be fused since it must be performed over integer tensors and
        it combines different elements of the input tensors.

        Returns:
            bool: False, this operation can not be fused as it adds different encrypted integers
        """
        return False


class QuantizedPad(QuantizedOp):
    """Quantized Padding op."""

    _impl_for_op_named: str = "Pad"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:

        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # Get the ONNX parameters
        self.mode = attrs.get("mode", None)
        assert_true(
            self.mode == "constant", "Padding operator only supports padding with a constant"
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Pad operation can not be fused since it must be performed over integer tensors.

        Returns:
            bool: False, this operation can not be fused as it is manipulates integer tensors
        """
        return False


class QuantizedWhere(QuantizedOp):
    """Where operator on quantized arrays.

    Supports only constants for the results produced on the True/False branches.
    """

    _impl_for_op_named: str = "Where"

    # This op takes a single variable input, so we can set int_input_names to a default id
    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # Remember that there are examples of where with more than 1 variable which are going to be
        # well managed, thanks to fusing: eg,
        # Act(x) = (x < 3) ? x + 42 : x ^ 42 would perfectly fuse
        # But for this kind of example, we have int_input_names = {'x'} since they all depend only
        # on x
        assert_true(
            self._int_input_names is not None and len(self._int_input_names) == 1,
            "internal issue: "
            + f"{self._int_input_names} "
            + "{len(self._int_input_names) if self._int_input_names is not None else -1}",
        )


class QuantizedCast(QuantizedOp):
    """Cast the input to the required data type.

    In FHE we only support a limited number of output types. Booleans are cast to integers.
    """

    _impl_for_op_named: str = "Cast"


class QuantizedGreater(QuantizedOp):
    """Comparison operator >.

    Only supports comparison with a constant.
    """

    _impl_for_op_named: str = "Greater"

    # Since this op takes a single variable input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support testing a > b where a,b are encrypted
        # only comparing to a constant is supported
        assert_true(constant_inputs is not None and len(constant_inputs) >= 1)


class QuantizedGreaterOrEqual(QuantizedOp):
    """Comparison operator >=.

    Only supports comparison with a constant.
    """

    _impl_for_op_named: str = "GreaterOrEqual"

    # Since this op takes a single variable input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support testing a >= b where a,b are encrypted
        # only comparing to a constant is supported
        assert_true(constant_inputs is not None and len(constant_inputs) >= 1)


class QuantizedLess(QuantizedOp):
    """Comparison operator <.

    Only supports comparison with a constant.
    """

    _impl_for_op_named: str = "Less"

    # Since this op takes a single variable input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support testing a < b where a,b are encrypted
        # only comparing to a constant is supported
        assert_true(constant_inputs is not None and len(constant_inputs) >= 1)


class QuantizedLessOrEqual(QuantizedOp):
    """Comparison operator <=.

    Only supports comparison with a constant.
    """

    _impl_for_op_named: str = "LessOrEqual"

    # Since this op takes a single variable input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support testing a <= b where a,b are encrypted
        # only comparing to a constant is supported
        assert_true(constant_inputs is not None and len(constant_inputs) >= 1)


class QuantizedOr(QuantizedOp):
    """Or operator ||.

    This operation is not really working as a quantized operation. It just works when things got
    fused, as in eg Act(x) = x || (x + 42))
    """

    _impl_for_op_named: str = "Or"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support Or between encrypted tensors
        # Only Or between
        # - encrypted tensors and float constants
        # - tensors that are produced by a unique integer tensor
        # is supported
        # Or between two constants is possible but should be optimized out by
        # the constant folding procedure
        assert_true(self.can_fuse() or (constant_inputs is not None and len(constant_inputs) == 1))

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Or can be fused and computed in float when a single integer tensor generates
        both the operands. For example in the formula: f(x) = x || (x + 1)  where x is an integer
        tensor.

        Returns:
            bool: Can fuse
        """
        return len(self._int_input_names) == 1


class QuantizedDiv(QuantizedOp):
    """Div operator /.

    This operation is not really working as a quantized operation. It just works when things got
    fused, as in eg Act(x) = 1000 / (x + 42))
    """

    _impl_for_op_named: str = "Div"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support Div between encrypted tensors
        # Only Div between
        # - encrypted tensors and float constants
        # - tensors that are produced by a unique integer tensor
        # is supported
        # Div between two constants is possible but should be optimized out by
        # the constant folding procedure
        assert_true(self.can_fuse() or (constant_inputs is not None and len(constant_inputs) == 1))

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Div can be fused and computed in float when a single integer tensor generates
        both the operands. For example in the formula: f(x) = x / (x + 1)  where x is an integer
        tensor.

        Returns:
            bool: Can fuse
        """
        return len(self._int_input_names) == 1


class QuantizedMul(QuantizedOp):
    """Multiplication operator.

    Only multiplies an encrypted tensor with a float constant for now. This operation will
    be fused to a (potentially larger) TLU.
    """

    _impl_for_op_named: str = "Mul"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # We do not support multiplication between encrypted tensors
        # Only multiplication between
        # - encrypted tensors and float constants
        # - tensors that are produced by a unique integer tensor
        # is supported
        # Multiplication between two constants is possible but should be optimized out by
        # the constant folding procedure
        assert_true(self.can_fuse() or (constant_inputs is not None and len(constant_inputs) == 1))

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Multiplication can be fused and computed in float when a single integer tensor generates
        both the operands. For example in the formula: f(x) = x * (x + 1)  where x is an integer
        tensor.

        Returns:
            bool: Can fuse
        """
        return len(self._int_input_names) == 1


class QuantizedSub(QuantizedAdd):
    """Subtraction operator.

    This works the same as addition on both encrypted - encrypted and on encrypted - constant.
    """

    _impl_for_op_named: str = "Sub"
    b_sign: int = -1


class QuantizedBatchNormalization(QuantizedOp):
    """Quantized Batch normalization with encrypted input and in-the-clear normalization params."""

    _impl_for_op_named: str = "BatchNormalization"


class QuantizedFlatten(QuantizedOp):
    """Quantized flatten for encrypted inputs."""

    _impl_for_op_named: str = "Flatten"

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Flatten the input integer encrypted tensor.

        Args:
            q_inputs: an encrypted integer tensor at index 0
            attrs: contains axis attribute

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

        assert_true(len(q_inputs) == 1, "Flatten operator only takes a single input")

        axis = attrs["axis"]
        newshape: Sequence[SupportsIndex]
        newshape = (
            *q_inputs[0].qvalues.shape[0:axis],
            numpy.prod(q_inputs[0].qvalues.shape[axis:]),
        )

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            prepared_inputs[0].quantizer.n_bits,
            numpy.reshape(prepared_inputs[0].qvalues, newshape),
            value_is_float=False,
            options=prepared_inputs[0].quantizer.quant_options,
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Flatten operation can not be fused since it must be performed over integer tensors.

        Returns:
            bool: False, this operation can not be fused as it is manipulates integer tensors.
        """

        return False


class QuantizedReduceSum(QuantizedOp):
    """ReduceSum with encrypted input."""

    _impl_for_op_named: str = "ReduceSum"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        """Construct the quantized ReduceSum operator and retrieve parameters.

        Args:
            n_bits_output (int): Number of bits for the operator's quantization of outputs.
            int_input_names (Optional[Set[str]]): Names of input integer tensors. Default to None.
            constant_inputs (Optional[Dict]): Input constant tensor.
                axes (Optional[numpy.ndarray]): Array of integers along which to reduce.
                    The default is to reduce over all the dimensions of the input tensor if
                    'noop_with_empty_axes' is false, else act as an Identity op when
                    'noop_with_empty_axes' is true. Accepted range is [-r, r-1] where
                    r = rank(data). Default to None.
            input_quant_opts (Optional[QuantizationOptions]): Options for the input quantizer.
                Default to None.
            attrs (dict): RecuseSum options.
                keepdims (int): Keep the reduced dimension or not, 1 means keeping the
                    input dimension, 0 will reduce it along the given axis. Default to 1.
                noop_with_empty_axes (int): Defines behaviour if 'axes' is empty or set to None.
                    Default behaviour with 0 is to reduce all axes. When axes is empty and this
                    attribute is set to true 1, input tensor will not be reduced, and the output
                    tensor would be equivalent to input tensor. Default to 0.
        """

        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # This attribute is truly set when calling the calibrate method
        self.input_shapes = None

        # Retrieve and set the ONNX parameters
        self.keepdims = attrs.get("keepdims", 1)
        self.noop_with_empty_axes = attrs.get("noop_with_empty_axes", 0)

        assert_true(
            self.keepdims == 0, "ReduceSum currently only outputs reduced dimension tensors."
        )

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Sum the encrypted tensor's values over axis 1.

        Args:
            q_inputs (QuantizedArray): An encrypted integer tensor at index 0.
            attrs (Dict): Contains axis attribute.

        Returns:
            (QuantizedArray): The sum of all values along axis 1 as an encrypted integer tensor.
        """

        assert_true(len(q_inputs) == 1, "ReduceSum only takes a single input")

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        axes = prepared_inputs[1]
        assert_true(
            axes is not None and len(axes.values.shape) == 1 and axes.values[0] == 1,
            "ReduceSum currently only handles summing over axis 1.",
        )

        input_qarray = prepared_inputs[0].qvalues
        assert_true(
            len(input_qarray.shape) == 2,
            "ReduceSum currently only handles arrays of 2 dimensions",
        )

        n_values = input_qarray.shape[1]
        assert_true(
            (n_values != 0) and (n_values & (n_values - 1) == 0),
            "ReduceSum only handles N values with N a power of 2.",
        )

        # Computing the total depth of the tree
        total_depth = int(numpy.log2(n_values))

        # Applying the tree sum technique with only one MSB.
        # Since our accumulators have limited bitwidth, it is impossible to sum many integers of a
        # certain precision (number of bits used) using numpy.sum() without overflowing. Therefore,
        # the idea is first to sum all pairs of integers (precision: p + p = p+1) and then divide
        # (floor division) the result by 2 (giving a precision of p). Multiplying this result by 2
        # later gives the original sum, minus a bit of error (0 or 1). We then do this step again
        # to each results until we get only one integer (MSB, for Most Valuable Bits), creating
        # an inversed binary tree (for n_values a power of 2). Therefore, the final sum can be
        # found by computing MSB*(2**total_depth) = MSB*n_values. The total amount of error caused
        # by this technique can reach up to total_depth*(n_values)/2 and is independent of the
        # actual integers' precision. However, the method doesn't properly work with
        # non-uniformally distributed input values, causing the MSB to vanish before the
        # reconstruction of the sum.
        for depth in range(total_depth, 0, -1):
            step = 2**depth
            sum_arr = input_qarray[:, 0:step:2] + input_qarray[:, 1 : step + 1 : 2]
            input_qarray = sum_arr // 2

        # Dequantize the values to float
        quantizer = prepared_inputs[0].quantizer
        scaled_msbs = quantizer.scale * (input_qarray[:, 0] + -(quantizer.zero_point))

        # Reconstruct the total sum. We need to keep the same range between the dequantized
        # values and the calibrated float computations
        # Note: 2**total_depth = n_values
        final_sum = scaled_msbs * n_values

        output_qarray = QuantizedArray(
            self.n_bits,
            final_sum,
            value_is_float=True,
            options=self.input_quant_opts,
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )
        return output_qarray


class QuantizedErf(QuantizedOp):
    """Quantized erf op."""

    _impl_for_op_named: str = "Erf"


class QuantizedNot(QuantizedOp):
    """Quantized Not op."""

    _impl_for_op_named: str = "Not"
