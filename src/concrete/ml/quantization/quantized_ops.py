"""Quantized versions of the ONNX operators for post training quantization."""

# pylint: disable=too-many-lines

# FIXME: #1018

from typing import Any, Dict, Optional, Sequence, Set, Union

import numpy
from concrete.onnx import conv as cnp_conv
from concrete.onnx import maxpool as cnp_maxpool
from typing_extensions import SupportsIndex

from ..common.debugging import assert_false, assert_true
from ..onnx.onnx_impl_utils import (
    compute_onnx_pool_padding,
    numpy_onnx_pad,
    onnx_avgpool_compute_norm_const,
)
from .base_quantized_op import QuantizedMixingOp, QuantizedOp, QuantizedOpUnivariateOfEncrypted
from .quantizers import QuantizationOptions, QuantizedArray, UniformQuantizationParameters


def _check_op_input_zero_point(zero_point: Any, op_name: Optional[str]):
    """Check that an operation's quantized input zero-point is a single value.

    Args:
        zero_point (Any): The input zero-point
        op_name (str): The name of the operation that is checking its input

    """

    # Checks with assert to also to ensure type safety
    assert zero_point is not None and (
        isinstance(zero_point, (int, float))
        or numpy.isscalar(zero_point)
        or (isinstance(zero_point, numpy.ndarray) and zero_point.size == 1)
    ), (
        f"Operation {op_name} is trying to use an input with a zero-point that is "
        "not a single value. Only model output quantizers can have zero-points that are arrays. "
    )


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


class QuantizedPow(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Quantized pow op.

    Only works for a float constant power. This operation will be fused to a (potentially
    larger) TLU.
    """

    _impl_for_op_named: str = "Pow"


class QuantizedGemm(QuantizedMixingOp):
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
        q_bias: Optional[numpy.ndarray] = (
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
        # in float and will thus be fused with any float processing that follows this layer.

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

            # Last part that has to be done in integer
            numpy_q_out = matmul + sum_input
        else:
            numpy_q_out = matmul

        # sum_weights is a constant
        sum_weights = q_input.quantizer.zero_point * numpy.sum(
            weights_q_values, axis=0, keepdims=True
        )

        final_term = p * q_input.quantizer.zero_point * q_weights.quantizer.zero_point

        # Note that here we do not rescale to the output_scale and we do not add a zero-point
        # Any following Gemm/MatMul/Conv layers will do the rescaling (during requantization)
        # by calling _prepare_inputs_with_constants(...quantize_real_values=True)
        m_matmul = q_input.quantizer.scale * q_weights.quantizer.scale

        # If this operation's result are network outputs, return
        # directly the integer values and a appropriate quantization parameters that
        # allow direct in-the-clear dequantization, including the bias
        if self.produces_graph_output:
            out_zp: Union[int, numpy.ndarray] = sum_weights - final_term
            if q_bias is not None:
                # Make mypy happy
                assert q_bias is not None
                # Reshape the biases to broadcast them to each neuron
                out_zp = out_zp + q_bias / (-m_matmul)

            # FIXME: remove when CNP fixes the return type of cnp_conv in non-tracing mode
            # https://github.com/zama-ai/concrete-numpy-internal/issues/1739
            if isinstance(numpy_q_out, numpy.ndarray):
                numpy_q_out = numpy_q_out.astype(numpy.int64)

            # We identify terms in the above equation to determine what
            # the scale/zero-point of the in-the-clear quantizer should be
            # to properly dequantize numpy_q_out
            return self.make_output_quant_parameters(numpy_q_out, m_matmul, out_zp)

        # Quantization scales and zero points (FLOATS involved)
        # This is going to be compiled with a PBS (along with the following activation function)
        numpy_q_out = numpy_q_out.astype(numpy.float64) + final_term - sum_weights

        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None:
            # The bias is handled as a float and will be fused
            numpy_q_out = numpy_q_out + q_bias

        # Return the float values, so that CN can fuse any following float operations
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
        # Moreover, the zero-point will be sum of input zero-points
        assert self.b_sign in [-1, 1]

        # This lines will be simplified into
        #       sum_q = rescale_q0 + self.b_sign * rescale_q1
        # when zama-ai/concrete-numpy-internal#1749 is done
        if self.b_sign == 1:
            sum_q = rescale_q0 + rescale_q1
        elif self.b_sign == -1:
            sum_q = rescale_q0 - rescale_q1

        # But we would like the output to have n_bits, so we dequantize
        dequant_sum = self.output_quant_params.scale * sum_q

        # Return the raw float values without re-quantizing them to the new scale, as any
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
            self.call_impl(prepared_inputs[0].qvalues, newshape, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )


class QuantizedConv(QuantizedMixingOp):
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
                dilations (Tuple[int]): dilation of the kernel. Default to 1 on all dimensions.
                group (int): number of convolution groups. Default to 1.
                kernel_shape (Tuple[int]): shape of the kernel. Should have 2 elements for 2d conv
                pads (Tuple[int]): padding in ONNX format (begin, end) on each axis
                strides (Tuple[int]): stride of the convolution on each axis
        """

        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # Get the ONNX parameters
        self.group = attrs.get("group", 1)
        self.kernel_shape = attrs.get("kernel_shape", None)
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

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
            len(self.pads) == 2 * len(self.kernel_shape),
            "The convolution operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard",
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

        in_channels = q_input.values.shape[1]
        weight_channels = q_weights.values.shape[1]
        assert_true(
            weight_channels == in_channels / self.group,
            f"Expected number of channels in weight to be {in_channels / self.group} "
            f"(C / group). Got {weight_channels}.",
        )

        out_channels = q_weights.values.shape[0]
        assert_true(
            out_channels % self.group == 0,
            f"Expected number of output channels O ({out_channels}) to be a multiple of "
            f"group ({self.group}).",
        )

        # Prepare a constant tensor to compute the sum of the inputs
        q_weights_1 = numpy.ones_like(q_weights.qvalues)

        assert q_weights.quantizer.scale is not None
        assert q_weights.quantizer.zero_point is not None

        assert q_input.quantizer.scale is not None
        assert q_input.quantizer.zero_point is not None

        # Can only pad with scalar zero-points, but zero-points can be float in special cases
        # for output layers
        _check_op_input_zero_point(q_input.quantizer.zero_point, self.op_instance_name)
        pad_value = int(q_input.quantizer.zero_point)
        q_input_pad = numpy_onnx_pad(q_input.qvalues, self.pads, pad_value, True)

        # We follow the Quantized Gemm implementation
        # which in turn follows Eq.7 in https://arxiv.org/abs/1712.05877
        # to split the core computation from the zero points and scales.

        # Compute the first encrypted term that convolves weights and inputs
        # Force padding to 0 as padding needs to use a custom padding initializer
        # and is thus manually performed in the code above
        fake_pads = [0] * len(self.pads)
        conv_wx = cnp_conv(
            q_input_pad,
            q_weights.qvalues,
            bias=None,
            pads=fake_pads,
            strides=self.strides,
            dilations=self.dilations,
            group=self.group,
        )

        # The total number of elements that are convolved by the application of a single kernel
        n_weights = numpy.prod(q_weights.qvalues.shape[1:])

        # If the weights have symmetric quantization, their zero point will be 0
        # The following check avoids the computation of the sum of the inputs, which may have
        # large bitwidth, in the case where it would be multiplied by zero
        if q_weights.quantizer.zero_point != 0:
            # Compute the sum of the inputs (second encrypted term)
            assert_true(
                isinstance(q_weights.quantizer.zero_point, (int, numpy.int_)),
                f"Zero point of weights tensor in {self.op_type} "
                f"op {self.op_instance_name} must be integer",
            )
            zw_conv_1x = -q_weights.quantizer.zero_point * cnp_conv(
                q_input_pad,
                q_weights_1,
                bias=None,
                pads=[0, 0, 0, 0],
                strides=self.strides,
                dilations=self.dilations,
                group=self.group,
            )
            numpy_q_out = conv_wx + zw_conv_1x
        else:
            numpy_q_out = conv_wx

        # Compute the third term, the sum of the weights which is a constant
        sum_weights = q_input.quantizer.zero_point * numpy.sum(
            q_weights.qvalues, axis=(1, 2, 3), keepdims=True
        ).transpose(1, 0, 2, 3)

        # Compute the forth term which is a constant
        final_term = n_weights * q_input.quantizer.zero_point * q_weights.quantizer.zero_point

        # Compute the rescaling factor that dequantizes the input
        # This is going to be compiled with a PBS (along with the following activation function)
        # Note that we don't requantize the output of the conv, this will be done by
        # any Gemm/Add/Conv layers that follow
        m_matmul = q_input.quantizer.scale * q_weights.quantizer.scale

        # If this operation's result are network outputs, return
        # directly the integer values and an appropriate quantization parameters that
        # allow direct in-the-clear dequantization, including the bias
        if self.produces_graph_output:
            # Note that to use the bias, we need to rescale it to the output scale
            # For Eq. 7 in  https://arxiv.org/abs/1712.05877, we can write:
            # S_out(q_out - zp_out) = S_x * S_w (multisum + bias / (S_x * S_w))
            # where multisum is the dot product of quantized inputs and quantized weights
            # Then we identify terms:
            #   S_out = S_x * S_w
            #   q_out = multisum terms involving inputs
            #   zp_out = -(multisum terms involving weights + bias / (S_x * S_w))
            out_zp: Union[int, numpy.ndarray] = sum_weights - final_term
            if q_bias is not None:
                # Reshape the biases to broadcast them to each channel
                out_zp = out_zp - q_bias.reshape((1, -1, 1, 1)) / m_matmul

            # FIXME: remove when CNP fixes the return type of cnp_conv in non-tracing mode
            # https://github.com/zama-ai/concrete-numpy-internal/issues/1739
            if isinstance(numpy_q_out, numpy.ndarray):
                numpy_q_out = numpy_q_out.astype(numpy.int64)

            # We identify terms in the above equation to determine what
            # the scale/zero-point of the in-the-clear quantizer should be
            # to properly dequantize numpy_q_out
            return self.make_output_quant_parameters(numpy_q_out, m_matmul, out_zp)

        # Now compute the whole sum (sum of the four terms)
        numpy_q_out = numpy_q_out.astype(numpy.float64) + final_term - sum_weights

        # Rescale from scale=scale_inputs x scale_outputs to output scale
        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None:
            # The bias addition is handled in float and will be fused into a TLU
            # Reshape the biases to broadcast them to each channel
            numpy_q_out = numpy_q_out + q_bias.reshape((1, -1, 1, 1))  # bias_part

        # And return as a QuantizedArray initialized from the float data, keeping
        # track of the quantization parameters
        return QuantizedArray(
            self.n_bits,
            numpy_q_out,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )


class QuantizedAvgPool(QuantizedMixingOp):
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
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

        # Validate the parameters
        assert_true(
            len(self.kernel_shape) == 2, "The Average Pool operator currently supports only 2d"
        )
        assert_true(
            len(self.kernel_shape) == len(self.strides),
            "The Average Pool operator requires the number of strides to "
            "be the same as the number of kernel dimensions",
        )
        assert_true(
            len(self.pads) == 2 * len(self.kernel_shape),
            "The Average Pool operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard",
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

        norm_const = 1.0 / onnx_avgpool_compute_norm_const(
            q_input.qvalues.shape, self.kernel_shape, self.pads, self.strides, self.ceil_mode
        )

        # for mypy: The Quantized ops can only run on QuantizedArray that have quantization
        # parameters (i.e. were fully constructed). This should always be the case, except
        # during the UniformQuantizer initialization when the zero_point can exist as None
        assert q_input.quantizer.zero_point is not None

        # Compute padding with floor and apply it to the input, pad with the input zero-point
        pool_pads = compute_onnx_pool_padding(
            q_input.qvalues.shape, self.kernel_shape, self.pads, self.strides, 0
        )

        # Can only pad with scalar zero-points, but zero-points can be float in special cases
        # for output layers
        _check_op_input_zero_point(q_input.quantizer.zero_point, self.op_instance_name)
        pad_value = int(q_input.quantizer.zero_point)
        q_input_pad = numpy_onnx_pad(q_input.qvalues, pool_pads, pad_value, int_only=True)

        if self.ceil_mode == 1:
            # Padding for tensorflow style

            # Compute padding with ceil and apply it to the input, pad with zeros, the zeros
            # will be ignored in the computation
            pool_pads_ceil = compute_onnx_pool_padding(
                q_input.qvalues.shape, self.kernel_shape, self.pads, self.strides, 1
            )

            # Can only pad with scalar zero-points, but zero-points can be float in special cases
            # for output layers
            q_input_pad_ceil = numpy_onnx_pad(q_input.qvalues, pool_pads_ceil, 0, True)

            # Copy the PyTorch style padded input to the larger 0 padded tensor
            q_input_pad_ceil[:, :, 0 : q_input_pad.shape[2], 0 : q_input_pad.shape[3]] = q_input_pad
            q_input_pad = q_input_pad_ceil

        # Remark that here, we are _not_ using Concrete-Numpy pad, since it would pad with
        # 0's while we want to pad with zero-point's. So, instead, he have done the padding
        # on our side, with q_input_pad
        fake_pads = [0] * len(self.pads)
        sum_result = cnp_conv(q_input_pad, kernel, None, fake_pads, self.strides)

        result = (
            sum_result.astype(numpy.float64) * norm_const - q_input.quantizer.zero_point
        ) * q_input.quantizer.scale

        return QuantizedArray(
            self.n_bits,
            result,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )


class QuantizedMaxPool(QuantizedOp):
    """Quantized Max Pooling op."""

    _impl_for_op_named: str = "MaxPool"

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
        self.auto_pad = attrs.get("auto_pad", "NOTSET")
        self.ceil_mode = attrs.get("ceil_mode", 0)
        self.kernel_shape = attrs.get("kernel_shape", None)
        self.storage_order = attrs.get("storage_order", 0)
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

        # Validate the parameters
        assert_true(
            self.ceil_mode == 0, "Only ceil_mode = 0 is supported by Concrete Numpy for now"
        )

        # Validate the parameters
        assert_true(len(self.kernel_shape) == 2, "The Max Pool operator currently supports only 2d")
        assert_true(
            len(self.kernel_shape) == len(self.strides),
            "The Max Pool operator requires the number of strides to "
            "be the same as the number of kernel dimensions",
        )
        assert_true(
            len(self.pads) == 2 * len(self.kernel_shape),
            "The Max Pool operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard",
        )

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

        # for mypy: The Quantized ops can only run on QuantizedArray that have quantization
        # parameters (i.e. were fully constructed). This should always be the case, except
        # during the UniformQuantizer initialization when the zero_point can exist as None
        assert q_input.quantizer.zero_point is not None

        assert_true(
            self.ceil_mode == 0, "Only ceil_mode = 0 is supported by Concrete Numpy for now"
        )

        # Simple padding for PyTorch style
        pool_pads = compute_onnx_pool_padding(
            q_input.qvalues.shape, self.kernel_shape, self.pads, self.strides, self.ceil_mode
        )

        q_input_pad = numpy_onnx_pad(
            q_input.qvalues, pool_pads, q_input.quantizer.zero_point, int_only=True
        )

        # Remark that here, we are _not_ using Concrete-Numpy pad, since it would pad with
        # 0's while we want to pad with zero-point's. So, instead, he have done the padding
        # on our side, with q_input_pad
        fake_pads = [0] * len(self.pads)
        sum_result = cnp_maxpool(
            q_input_pad,
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            auto_pad=self.auto_pad,
            pads=fake_pads,
            dilations=self.dilations,
            ceil_mode=self.ceil_mode,
            storage_order=self.storage_order,
        )

        result = (
            sum_result.astype(numpy.float64) - q_input.quantizer.zero_point
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

        Max Pooling operation can not be fused since it must be performed over integer tensors and
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

        Pad operation cannot be fused since it must be performed over integer tensors.

        Returns:
            bool: False, this operation cannot be fused as it is manipulates integer tensors
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


class QuantizedOr(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Or operator ||.

    This operation is not really working as a quantized operation. It just works when things got
    fused, as in e.g. Act(x) = x || (x + 42))
    """

    _impl_for_op_named: str = "Or"


class QuantizedDiv(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Div operator /.

    This operation is not really working as a quantized operation. It just works when things got
    fused, as in e.g. Act(x) = 1000 / (x + 42))
    """

    _impl_for_op_named: str = "Div"


class QuantizedMul(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Multiplication operator.

    Only multiplies an encrypted tensor with a float constant for now. This operation will
    be fused to a (potentially larger) TLU.
    """

    _impl_for_op_named: str = "Mul"


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

        Flatten operation cannot be fused since it must be performed over integer tensors.

        Returns:
            bool: False, this operation cannot be fused as it is manipulates integer tensors.
        """

        return False


class QuantizedReduceSum(QuantizedMixingOp):
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
                noop_with_empty_axes (int): Defines behavior if 'axes' is empty or set to None.
                    Default behavior with 0 is to reduce all axes. When axes is empty and this
                    attribute is set to true 1, input tensor will not be reduced, and the output
                    tensor would be equivalent to input tensor. Default to 0.
        """
        super().__init__(n_bits_output, int_input_names, constant_inputs, input_quant_opts, **attrs)

        # Retrieve and set the ONNX parameters
        self.keepdims = attrs.get("keepdims", 1)
        self.noop_with_empty_axes = attrs.get("noop_with_empty_axes", 0)

    def calibrate(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            *inputs (numpy.ndarray): Calibration sample inputs.

        Returns:
            numpy.ndarray: The output values for the provided calibration samples.
        """
        # Here we need the actual values of the constants, we need to pass through
        # the numpy.ndarrays in the computation graph
        prepared_inputs = self._prepare_inputs_with_constants(
            *inputs, calibrate=True, quantize_actual_values=False
        )

        # Retrieve values and axes parameter
        values = prepared_inputs[0]
        axes = prepared_inputs[1]

        # As the calibration inputset and inputs are ran over several samples, we need to apply the
        # sum on all the given axes except the first one (the sample axis), including when axes is
        # set to None (i.e. sum over all axes).
        prepared_inputs[1] = (
            numpy.array(tuple(axes + 1)) if axes is not None else numpy.arange(1, len(values.shape))
        )

        quantized_samples = QuantizedArray(
            self.n_bits, self.call_impl(*prepared_inputs, **self.attrs)
        )

        self.output_quant_params = quantized_samples.quantizer.quant_params
        self.output_quant_stats = quantized_samples.quantizer.quant_stats

        return quantized_samples.values

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Sum the encrypted tensor's values along the given axes.

        Args:
            q_inputs (QuantizedArray): An encrypted integer tensor at index 0.
            attrs (Dict): Options are handled in constructor.

        Returns:
            (QuantizedArray): The sum of all values along the given axes.
        """
        # Retrieve the quantized inputs as well as the function's constant parameters
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # Retrieve values and axes parameter
        q_values = prepared_inputs[0].qvalues
        axes = prepared_inputs[1]

        # As the calibration inputset and inputs are ran over several samples, we need to apply the
        # sum on all the given axes except the first one (the sample axis), including when axes is
        # set to None (i.e. sum over all axes).
        axes = (
            tuple(axes + 1)
            if axes is not None
            else tuple(axis for axis in range(1, len(q_values.shape)))
        )

        # Numpy's keepdims parameter is a boolean while ONNX's one is an int (0 or 1). Even though
        # Python handles them equivalently, we need to manually convert it as mypy doesn't accept
        # this type difference
        keepdims = bool(self.keepdims)

        # Sum all the quantized values
        q_sum = numpy.sum(q_values, axis=axes, keepdims=keepdims)

        # Determining the number of output zero_points to use for dequantization with the total
        # number of elements summed all together, which is the product of all the number of elements
        # found within the given axes
        n_elem = numpy.prod([q_values.shape[axis] for axis in axes])

        # Determining the output scale and zero_point
        input_quantizer = prepared_inputs[0].quantizer
        scale = input_quantizer.scale
        zero_point = n_elem * input_quantizer.zero_point

        # If this operator is the graph's last operator, there's is no need to created additional
        # TLUs
        if self.produces_graph_output:
            return self.make_output_quant_parameters(q_sum, scale, zero_point)

        # Dequantize the sum
        f_sum = scale * (q_sum - zero_point)

        sum_qarray = QuantizedArray(
            self.n_bits,
            f_sum,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

        return sum_qarray


class QuantizedErf(QuantizedOp):
    """Quantized erf op."""

    _impl_for_op_named: str = "Erf"


class QuantizedNot(QuantizedOp):
    """Quantized Not op."""

    _impl_for_op_named: str = "Not"


class QuantizedBrevitasQuant(QuantizedOp):
    """Brevitas uniform quantization with encrypted input."""

    _impl_for_op_named: str = "onnx.brevitas.Quant"
    # FIXME: Note that this should be reset when the correctness test that finds
    # all mismatches between CML and Brevitas is fixed
    # https://github.com/zama-ai/concrete-ml-internal/issues/2373
    quantize_inputs_with_model_outputs_precision = True
    output_quant_opts: QuantizationOptions

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

        self.is_signed = bool(attrs["signed"])
        self.is_narrow = bool(attrs["narrow"])

        assert_false(
            not self.is_signed and self.is_narrow,
            "Can not use narrow range for non-signed Brevitas quantizers",
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

        # For mypy
        assert constant_inputs is not None

        # The constant inputs can have either int or str keys, here it's int
        n_bits = constant_inputs[3]  # type: ignore

        # Set the QAT flag on the output of this operation, so that the
        # following operation in the graph is aware of this flag and can
        # just copy the quantization
        self.output_quant_opts = self._get_output_quant_opts()
        self.output_quant_opts.n_bits = n_bits
        self.output_quant_opts.is_qat = True

        # Disable the QAT value checker as we have the true parameters in ONNX
        # Brevitas quantization layers store the scale/zero_point in the ONNX file
        # so we don't need to compute/infer them
        self.output_quant_opts.is_precomputed_qat = True
        self.output_quant_opts.is_narrow = self.is_narrow
        self.output_quant_opts.is_signed = self.is_signed

    def calibrate(self, *inputs: numpy.ndarray) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of Quantization function.

        Args:
            *inputs (numpy.ndarray): Calibration sample inputs.

        Returns:
            numpy.ndarray: the output values for the provided calibration samples.
        """

        result = super().calibrate(*inputs)

        # Override the output quantization params with
        # those stored in the ONNX model. This allows the quantized module to
        # pass these parameters to the module's input quantizer

        # for mypy: this is set by the base class calibration
        assert self.output_quant_params is not None

        self.output_quant_params = UniformQuantizationParameters()
        self.output_quant_params.scale = numpy.float64(self.constant_inputs[1])
        self.output_quant_params.zero_point = int(self.constant_inputs[2])

        n_bits = int(self.constant_inputs[3])
        self.output_quant_params.offset = 2 ** (n_bits - 1) if self.is_signed else 0

        return result

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
        n_bits = int(prepared_inputs[3])

        assert len(q_inputs) >= 1

        assert_true(
            self.output_quant_params is not None, "You need to calibrate this op before using it"
        )

        # For mypy
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None

        # Pass-through when the input is already quantized in the manner that this
        # layer wants to quantize it. This is done to optimize out the model input TLU.
        # Only works when the layer has been calibrated.
        if (
            q_inputs[0].quantizer.quant_options.is_equal(self.output_quant_opts)
            and q_inputs[0].quantizer.quant_params.scale == self.output_quant_params.scale
            and q_inputs[0].quantizer.quant_params.zero_point == self.output_quant_params.zero_point
        ):
            return q_inputs[0]

        f_outputs = self.call_impl(*prepared_inputs, **attrs)
        res = QuantizedArray(
            n_bits,
            f_outputs,
            value_is_float=True,
            options=self.output_quant_opts,
            stats=self.output_quant_stats,
            params=self.output_quant_params,
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

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )


class QuantizedFloor(QuantizedOp):
    """Quantized Floor op."""

    _impl_for_op_named: str = "Floor"


class QuantizedMax(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Quantized Max op."""

    _impl_for_op_named: str = "Max"


class QuantizedMin(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Quantized Min op."""

    _impl_for_op_named: str = "Min"


class QuantizedNeg(QuantizedOp):
    """Quantized Neg op."""

    _impl_for_op_named: str = "Neg"


class QuantizedSign(QuantizedOp):
    """Quantized Neg op."""

    _impl_for_op_named: str = "Sign"


class QuantizedUnsqueeze(QuantizedOp):
    """Unsqueeze operator."""

    _impl_for_op_named: str = "Unsqueeze"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Unsqueeze the input tensors on a given axis.

        Args:
            q_inputs: an encrypted integer tensor
            attrs: additional optional unsqueeze options

        Returns:
            result (QuantizedArray): unsqueezed encrypted integer tensor
        """

        # FIXME: Currently Unsqueeze quantizes the inputs, but this is unnecessary if the reshape
        # operation could be fused into a Gemm/Add/Conv that follows it. We should concatenate
        # here only if the concatenated result is returned directly from the FHE program.
        # See https://github.com/zama-ai/concrete-ml-internal/issues/527
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        axes = prepared_inputs[1]
        assert_true(isinstance(axes, list))

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, axes, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )


class QuantizedConcat(QuantizedOp):
    """Concatenate operator."""

    _impl_for_op_named: str = "Concat"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(self, *q_inputs: QuantizedArray, **attrs) -> QuantizedArray:
        """Concatenate the input tensors on a giver axis.

        Args:
            q_inputs: an encrypted integer tensor
            attrs: additional optional concatenate options

        Returns:
            result (QuantizedArray): concatenated encrypted integer tensor
        """

        # FIXME: Currently Concatenate quantizes the inputs, but this is unnecessary if the reshape
        # operation could be fused into a Gemm/Add/Conv that follows it. We should concatenate
        # here only if the concatenated result is returned directly from the FHE program.
        # See https://github.com/zama-ai/concrete-ml-internal/issues/527
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # The input tensors must have the same quantization parameters to be concatenated.
        scales = [x.quantizer.scale for x in prepared_inputs]
        zero_points = [x.quantizer.zero_point for x in prepared_inputs]
        assert_true(
            all(x == scales[0] for x in scales) and all(x == zero_points[0] for x in zero_points),
            "All inputs must have the same scale and zero_point to be concatenated.",
        )

        tensors_to_concat = [prepared_input.qvalues for prepared_input in prepared_inputs]

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(*tensors_to_concat, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )
