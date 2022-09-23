"""Quantized versions of the ONNX operators for post training quantization."""

# pylint: disable=too-many-lines
# FIXME: #1018

from typing import Any, Dict, Optional, Sequence, Set, SupportsIndex, Union

import numpy
from concrete.onnx import conv as cnp_conv

from ..common.debugging import assert_true
from .base_quantized_op import QuantizedOp
from .quantizers import QuantizationOptions, QuantizedArray, UniformQuantizationParameters


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

        # Core matmul operation in full integers with a shape change (INTEGERS)
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
            numpy_q_out = numpy_q_out + q_bias

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

        # Dequantize with input params and requantize with output parameters
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

        # But we would like the output to have n_bits, so we dequantize
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
    quantize_inputs_with_model_outputs_precision = True

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

        newshape = prepared_inputs[1]
        assert_true(numpy.issubdtype(newshape.dtype, numpy.integer))

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
        q_bias: Optional[numpy.ndarray] = None if len(prepared_inputs) == 2 else prepared_inputs[2]

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
            numpy_q_out = numpy_q_out + q_bias.reshape((1, -1, 1, 1))  # bias_part

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

        assert_true(
            self.constant_inputs is None or bool(numpy.all(self.constant_inputs[1] == 0)),
            "Padding operator supported only with all pads at zero",
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
        # well managed, thanks to fusing:
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
    fused, as in e.g. Act(x) = x || (x + 42))
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
    fused, as in e.g. Act(x) = 1000 / (x + 42))
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
    quantize_inputs_with_model_outputs_precision = True

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
    """ReduceSum with encrypted input.

    This operator is currently an experimental feature.
    """

    _impl_for_op_named: str = "ReduceSum"
    quantize_inputs_with_model_outputs_precision = True
    n_values: int
    total_depth: int

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
                noop_with_empty_axes (int): Defines behavior if 'axes' is empty or set to None.
                    Default behavior with 0 is to reduce all axes. When axes is empty and this
                    attribute is set to true 1, input tensor will not be reduced, and the output
                    tensor would be equivalent to input tensor. Default to 0.
        """
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # This attribute is truly set when calling the calibrate method
        self.input_shapes = None

        # Retrieve and set the ONNX parameters
        self.keepdims = attrs.get("keepdims", 1)
        self.noop_with_empty_axes = attrs.get("noop_with_empty_axes", 0)
        self.quantizers_: list = []
        assert_true(
            self.keepdims == 1,
            "ReduceSum currently only keeps the inputs' dimensions for its outputs.",
        )

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
        values = prepared_inputs[0]

        # Tree sum calibration has to be done over quantized values
        input_qarray = QuantizedArray(
            self.input_quant_opts.n_bits,
            values,
            options=self.input_quant_opts,
            stats=None,
            params=None,
        ).qvalues

        # Quantize input
        self.n_values = input_qarray.shape[1]
        assert_true(
            (self.n_values != 0) and (self.n_values & (self.n_values - 1) == 0),
            "ReduceSum only handles N values with N a power of 2.",
        )

        # Computing the total depth of the tree
        self.total_depth = int(numpy.log2(self.n_values))

        # Calibrate the tree sum
        self.tree_sum(input_qarray, True)

        return super().calibrate(*inputs)

    def tree_sum(self, input_qarray, is_calibration=False):
        """Large sum without overflow (only MSB remains).

        Args:
            input_qarray: Enctyped integer tensor.
            is_calibration: Whether we are calibrating the tree sum. If so, it will create all the
                quantizers for the downscaling.

        Returns:
            (numpy.ndarray): The MSB (based on the precision self.n_bits) of the integers sum.
        """

        # The method nables summing N integers without overflowing.
        # Since our accumulators have limited bitwidth (8 bits), it is impossible to sum too many
        # integers of a certain precision (synonym for bitwidth) without overflowing. For instance,
        # the sum numpy.sum([100, 60, 65, 59]) = 284 can't be executed in FHE as the bitwidth of
        # 284 is 9 > 8. Let k be an unsigned integer, it is possible to compute it's precision p
        # using floor(log2(k)) + 1, such that 2**(p-1) <= k < 2**p.

        # Let's now define two unsigned integers of precision p, a and b. By definition, the most
        # extreme reachable value is 2**p-1. The rest of this explanation will only consider worst
        # case scenarios as we want to ensure the operator's proper execution at any time. Let's
        # thus set a = b = 2**p-1.
        # The workaround starts by summing inputs by pairs. Then, the results are divided by 2
        # using the floor division in order to retrieve their quotients. Using the above notations,
        # we first have c = a + b = 2*(2**p-1) = 2**(p+1)-2. Since 2**p <= c < 2**(p+1), we obtain
        # that c is of precision p+1. Then, let d = floor(c/2) = floor(2**p-1) = 2**p-1 = a, making
        # d of precision p. The maximum bitwidth reached along these steps is therefore p+1,
        # meaning that, currently, they can be executed in FHE without any issues as long as
        # p+1 <= 8, which limits us to p <= 7.

        # Additionally, since d is of precision p, it is possible to sum it with another integer of
        # the same precision. In fact, more generally, we can apply the above steps to each pairs
        # of inputs, repeat the process with their outputs and then continue as such until we get
        # only one final integer (the sum's MSb, for Most Significant Bits). Considering that we
        # currently force the number of inputs (n_values) to be a power of 2 as we can't properly
        # handle other cases, this creates an inverted binary tree. Here's an example with
        # n_values=4 and p=3, using unsigned integers:

        # 0:   2   5 7   3
        #       \ /   \ /      (addition)
        #        7     10
        #        |     |       (division)
        # 1:     3     5
        #         \   /        (addition)
        #           8
        #           |          (division)
        # 2:        4

        # In this case, the true sum is 17, which uses 5 bits. The workaround estimates sum // n
        # with 4, a p-bit integer, without having any accumulators reach values above 4 bits of
        # precision.
        # Let d be the tree's total depth, meaning d = log2(n_values), we can then use that final
        # MSb m in order to approximate the true sum by computing m*(2**d) = m*n_values. The idea
        # behind this is that the true sum got divided by 2 at each of the tree' levels (d times in
        # total). In fact, this approximated sum is generally close to the true sum without exactly
        # matching it. In the above example, the approximation gives 4*4=16, resulting in an error
        # of 17-16=1. This is due to the fact that the division used in the process is the floor
        # division, which can create an error of 1 when applied to odd integers each time it is
        # called. More generally, the resulting sum approximates the true sum's p most
        # significant bits.

        # This error term (the division's remainder) is not accumulated during the process as it
        # could easily make the accumulators exceed their p-bit precision, especially with large
        # number of features. In fact, it is possible to exactly compute the maximum error reachable
        # by the workaround. Let l be a tree's level (with l=0 the upper level, representing the
        # inputs), the maximum amount of error accumulated in this level is the number of nodes
        # n_values/(2**l) times the remainders' weight 2**(l-1). Summing this over all of the
        # d levels, we get d*(n_values)/2.
        # Let S be the true sum, the approximated sum will therefore end within the range
        # [S - d*(n_values)/2, S]. It is important to not that this error term is of O(n*log(n))
        # and is independent of the actual inputs' precision. This means that currently, the more
        # features there are, the less precise the sum gets, no matter the input's precision.

        # In fact, the integer sum is done over signed integers because of how QuantizedOps are
        # built. This doesn't create any issues in most cases as it only applies an additional
        # linear transformation on the inputs' values without truly affecting the true sum. However,
        # we decided to use the numpy.rint operator
        # (https://numpy.org/doc/stable/reference/generated/numpy.rint.html) combined to
        # the true division instead of a single floor division in order to avoid problems with -1
        # values. By construction, the floor division gives the same results as a true division
        # followed by a floor operator. In particular, this means that -1//2 = -1, which is not
        # adapted to this workaround and can lead to a greater amount of errors. The main idea is
        # that a -1 value can be kept along many steps without vanishing if it gets summed with
        # zeros or small negative values. For example:

        # Using the floor division :
        # 0:  -1   0 0   0
        #       \ /   \ /      (addition)
        #       -1     0
        #        |     |       (division)
        # 1:    -1     0
        #         \   /        (addition)
        #          -1
        #           |          (division)
        # 2:       -1

        # In this case, the sum is approximated to -1*4 = -4 while the true sum is only -1.
        # More generally, an array with one -1 and (n-1) 0 would output -n while the true sum
        # remains to -1.
        # We therefore use numpy.rint's rounding properties, which rounds up .5 values to the
        # nearest even integer. This not only enables -0.5 values to be rounded up to 0 instead of
        # -1 but also partially compensates the global error by either "losing" or "earning"
        # remainders for each odd sum. However, this is not truly ideal. This makes the
        # workaround less intuitive, as it increases the overall error range by 2 (compared to the
        # floor division) and makes the accumulation of remainder errors less tractable. Also, all
        # remainder losses or earnings are not equivalent between each tree levels, as they don't
        # represent the same weights, making the notion of "average compensation" less viable. On
        # the other hand, we empirically observed that the rint operator leads to more accurate
        # results when used in practice with linear models. We therefore decided to keep it for now.

        # Note: We now use calibration. As such, the addition can be divided by any floating point
        # number between 1 and 2. If the number if below 2 then a zero_point is applied
        # to recalibrate the sum. This allows us much more precision as the distribution
        # of the addition often never completely fills the theoretical precision.

        for i, depth in enumerate(range(self.total_depth, 0, -1)):
            step = 2**depth
            sum_arr = input_qarray[:, 0:step:2] + input_qarray[:, 1 : step + 1 : 2]
            if is_calibration:
                # Compute calibration constants
                quantizer_ = QuantizedArray(n_bits=self.n_bits, values=sum_arr.astype(float))
                self.quantizers_.append(quantizer_)

            # Apply calibration
            input_qarray = self.quantizers_[i].update_values(sum_arr)
        return input_qarray

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
            axes is not None and len(axes.shape) == 1 and axes[0] == 1,
            "ReduceSum currently only handles summing over axis 1.",
        )

        input_qarray = prepared_inputs[0].qvalues
        assert_true(
            len(input_qarray.shape) == 2,
            "ReduceSum currently only handles arrays of 2 dimensions",
        )

        input_qarray = self.tree_sum(input_qarray)

        # Approximate the total sum. We need to keep the same range between the dequantized
        # values and the calibrated float computations
        # Note: 2**total_depth = n_values
        for depth, quantizer_ in enumerate(self.quantizers_[::-1]):
            input_qarray = quantizer_.quantizer.scale * (
                input_qarray - quantizer_.quantizer.zero_point * 2**depth
            )

        # Dequantize the values to float
        input_quantizer = prepared_inputs[0].quantizer
        scaled_msbs = input_quantizer.scale * (
            input_qarray + -(2**self.total_depth * input_quantizer.zero_point)
        )

        final_sum = scaled_msbs

        output_qarray = QuantizedArray(
            self.n_bits,
            final_sum,
            value_is_float=True,
            options=self._get_output_quant_opts(),
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


class QuantizedBrevitasQuant(QuantizedOp):
    """Brevitas uniform quantization with encrypted input."""

    _impl_for_op_named: str = "onnx.brevitas.Quant"

    def __init__(
        self,
        n_bits_output: int,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        """Construct the Brevitas quantization operator.

        Args:
            n_bits_output (int): Number of bits for the operator's quantization of outputs.
                Not used, will be overridden by the bit_width in ONNX
            int_input_names (Optional[Set[str]]): Names of input integer tensors. Default to None.
            constant_inputs (Optional[Dict]): Input constant tensor.
                scale (float): Quantizer scale
                zero_point (float): Quantizer zero-point
                bit_width (int): Number of bits of the integer representation
            input_quant_opts (Optional[QuantizationOptions]): Options for the input quantizer.
                Default to None.
            attrs (dict):
                rounding_mode (str): Rounding mode (default and only accepted option is "ROUND")
                signed (int): Whether this op quantizes to signed integers (default 1),
                narrow (int): Whether this op quantizes to a narrow range of integers
                    e.g. [-2**n_bits-1 .. 2**n_bits-1] (default 0),
        """

        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        def check_float(v, err_msg):
            assert_true(
                isinstance(v, float)
                or (isinstance(v, numpy.ndarray) and numpy.issubdtype(v.dtype, numpy.floating)),
                err_msg,
            )

        assert_true(
            attrs["rounding_mode"] == "ROUND",
            "Only rounding quantization is supported for Brevitas",
        )
        assert_true(
            int(attrs["signed"]) in {0, 1},
            "Signed flag in Brevitas quantizer must be 0/1",
        )
        assert_true(
            int(attrs["narrow"]) in {0, 1},
            "Narrow range flag in Brevitas quantizer must be 0/1",
        )

        # To ensure dequantization produces floats, the following parameters must be float.
        # This should be default export setting in Brevitas
        check_float(
            constant_inputs is not None
            and self.constant_inputs[self._params_name_to_input_idx["scale"]],
            "Scale of Brevitas Quant op must be float",
        )
        check_float(
            constant_inputs is not None
            and self.constant_inputs[self._params_name_to_input_idx["zero_point"]],
            "Zero Point of Brevitas Quant op must be float",
        )

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Quantize values.

        Args:
            q_inputs: an encrypted integer tensor at index 0 and one constant shape at index 1
            attrs: additional optional reshape options

        Returns:
            result (QuantizedArray): reshaped encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=False
        )

        # Impose the number of bits that this op quantizes its inputs, to the value that
        # was provided by Brevitas. This number of bits is normally
        # different from the op's n_bits. The op will usually have an output number of bits
        # that is suitable for network output, as the op could be the last op in the graph

        # Thus, we enforce the QAT number of bits on this op's output
        n_bits = prepared_inputs[3]

        # Copy the quantization parameters that were used in training
        quant_params = UniformQuantizationParameters()
        quant_params.scale = prepared_inputs[1]
        quant_params.zero_point = prepared_inputs[2]

        # FIXME: figure out if this check is needed
        # the code seems to run with narrow == 1
        # https://github.com/zama-ai/concrete-ml-internal/issues/1591
        # assert_true(
        #     self.attrs["signed"] == 1 and self.attrs["narrow"] == 0,
        #     "Only signed wide range Brevitas quantization giving 0 offset "
        #     "is implemented for now\n"
        #     f"{self.attrs['signed']=} ?=1 , {self.attrs['narrow']=} ?=0",
        # )

        # Currently we can not determine if symmetric quantization was used,
        # we just assume the offset is 0, the offset is not used for clipping in QAT.
        quant_params.offset = 0

        # Set the QAT flag on the output of this operation, so that the
        # following operation in the graph is aware of this flag and can
        # just copy the quantization
        options = self._get_output_quant_opts()
        options.is_qat = True

        f_outputs = self.call_impl(*prepared_inputs, **attrs)
        res = QuantizedArray(
            n_bits,
            f_outputs,
            value_is_float=True,
            options=options,
            stats=self.output_quant_stats,
            params=quant_params,
        )
        return res


class QuantizedTranspose(QuantizedOp):
    """Transpose operator for quantized inputs.

    This operator performs quantization, transposes the encrypted data, then
    dequantizes again.
    """

    _impl_for_op_named: str = "Transpose"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Reshape the input integer encrypted tensor.

        Args:
            q_inputs: an encrypted integer tensor at index 0 and one constant shape at index 1
            attrs: additional optional reshape options

        Returns:
            result (QuantizedArray): reshaped encrypted integer tensor
        """

        # FIXME: Currently Transpose quantizes the inputs, but this is unnecessary if the reshape
        # operation could be fused into a Gemm/Add/Conv that follows it. We should transpose
        # here only if the transpose result is returned directly from the FHE program.
        # See https://github.com/zama-ai/concrete-ml-internal/issues/527
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        axes_permute = self.attrs["perm"]

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            numpy.transpose(prepared_inputs[0].qvalues, axes_permute),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )
