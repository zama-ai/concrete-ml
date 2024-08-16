"""Quantized versions of the ONNX operators for post training quantization."""

# pylint: disable=too-many-lines

# This file is too long and should be split
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1018

from typing import Any, Dict, Optional, Sequence, Set, Union

import numpy
from concrete.fhe import conv as fhe_conv
from concrete.fhe import maxpool as fhe_maxpool
from concrete.fhe import tag, univariate, zeros
from typing_extensions import SupportsIndex

from ..common.debugging import assert_false, assert_true
from ..onnx.onnx_impl_utils import (
    compute_onnx_pool_padding,
    numpy_onnx_pad,
    onnx_avgpool_compute_norm_const,
)
from ..onnx.ops_impl import RawOpOutput
from .base_quantized_op import (
    ONNXOpInputOutputType,
    QuantizedMixingOp,
    QuantizedOp,
    QuantizedOpUnivariateOfEncrypted,
)
from .quantizers import (
    QuantizationOptions,
    QuantizedArray,
    UniformQuantizationParameters,
    UniformQuantizer,
)


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
        op_instance_name: str,
        int_input_names: Optional[Set[str]] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        alpha = self.attrs.get("alpha", 1)
        beta = self.attrs.get("beta", 1)

        assert_true(
            alpha == 1 and beta in [0, 1],
            f"{self.__class__.__name__} currently only supports alpha == 1 and beta in [0, 1].\n"
            f"Got alpha == {alpha} and beta == {beta}.",
        )

    @classmethod
    def supported_by_linear_backend(cls) -> bool:
        return True

    # pylint: disable-next=too-many-statements,too-many-locals
    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,
        **attrs,
    ) -> ONNXOpInputOutputType:

        alpha = self.attrs.get("alpha", 1)
        beta = self.attrs.get("beta", 1)

        # If self.constant_inputs is empty this is an encrypted gemm
        # There might be caveats here
        # (for example when one of the input is passed in clear with encrypted statuses.)
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4132
        is_encrypted_gemm = isinstance(self.constant_inputs, dict) and not self.constant_inputs

        # If alpha != 1 or beta not in [0, 1], this function must be modified
        assert_true(alpha == 1)
        assert_true(beta in {0, 1})

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input1 = prepared_inputs[0]
        assert isinstance(q_input1, QuantizedArray)
        q_input2 = prepared_inputs[1]
        assert isinstance(q_input2, QuantizedArray)

        # In the operation Y = alpha * A' * B' + beta * C, q_bias is used for
        # generalised matrix multiplication. q_bias is set to None for standard
        # matrix multiplication (beta == 0 or only two inputs)
        q_bias = None if len(prepared_inputs) == 2 or beta == 0 else prepared_inputs[2]
        assert isinstance(q_bias, (type(None), QuantizedArray))

        # Using snake case here to please the Python format, the original attrs don't have the '_'
        # Use default false so we also support MatMul impl, MatMul does not have these flags
        transpose_inputs1 = attrs.get("transA", False)
        with tag(self.op_instance_name + ".input"):
            input1_q_values = (
                numpy.transpose(q_input1.qvalues) if transpose_inputs1 else q_input1.qvalues
            )

        transpose_inputs2 = attrs.get("transB", False)
        input2_q_values = (
            numpy.transpose(q_input2.qvalues) if transpose_inputs2 else q_input2.qvalues
        )

        assert_true(
            input2_q_values.ndim in {2, 3},
            f"Unsupported dimension for the weight input of the gemm: {input2_q_values.ndim}",
        )

        # For mypy
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None

        assert q_input2.quantizer.scale is not None
        assert q_input2.quantizer.zero_point is not None

        assert q_input1.quantizer.scale is not None
        assert q_input1.quantizer.zero_point is not None

        # The following MatMul is done with integers, and thus, does not use of any PBS.
        # Rescaling the output of the integer MatMul to handle scale changes is done
        # in float and will thus be fused with any float processing that follows this layer.

        # Here we follow Eq.7 in https://arxiv.org/abs/1712.05877 to split the core computation
        # from the zero points and scales.

        p = input2_q_values.shape[-2]

        # Remove the manual matrix multiplication when we can handle input precision with rounding
        # FIXME: https://github.com/zama-ai/concrete-internal/issues/512
        def enc_mul(x, y):
            r"""Encrypted multiplication of two input arrays.

            This function computes the encrypted multiplication of two numpy arrays.
            It uses the following equality:

            \[
            (x + y)^2 - (x - y)^2 = 4xy
            \]

            This equation simplifies to the standard multiplication operation.

            In TFHE, this allows to do encrypted multiplication with 2 PBS.

            Args:
                x (numpy.ndarray): The first input numpy array.
                y (numpy.ndarray): The second input numpy array.

            Returns:
                numpy.ndarray: The result of the encrypted multiplication.
            """
            with tag("pbs_multiplication"):
                # Compute sum and difference of x and y
                add = x + y
                sub = x - y

                # Apply Concrete rounding to the addition and substraction
                with tag(self.op_instance_name + ".pbs_matmul_rounding_add"):
                    add = self.cnp_round(add, calibrate_rounding, rounding_operation_id="add")
                with tag(self.op_instance_name + ".pbs_matmul_rounding_sub"):
                    sub = self.cnp_round(sub, calibrate_rounding, rounding_operation_id="sub")

                # Square the rounded sums and differences, and divide by 4
                add_pow = (add.astype(numpy.float64)) ** 2
                sub_pow = (sub.astype(numpy.float64)) ** 2
                add_pow_divide = (add_pow / 4.0).astype(numpy.int64)
                sub_pow_divide = (sub_pow / 4.0).astype(numpy.int64)

            # Return the result of the multiplication
            return add_pow_divide - sub_pow_divide

        # Remove the manual matrix multiplication when we can handle input precision with rounding
        # FIXME: https://github.com/zama-ai/concrete-internal/issues/512
        def matmul(a, b):
            """Matrix multiplication of two input arrays, supporting 2D or 3D.

            This function performs matrix multiplication on either 2D or 3D numpy arrays.
            It supports batch processing, where either or both inputs can be a batch
            (3D array), and handles the reshaping and summation operations required
            for matrix multiplication.

            Args:
                a (numpy.ndarray): The first input array, can be 2D or 3D.
                b (numpy.ndarray): The second input array, can be 2D or 3D.

            Returns:
                numpy.ndarray: The result of the matrix multiplication.
            """
            with tag("encrypted_matmul"):
                # Determine the dimensions of inputs and handle 3D (batch) inputs
                a_3d = a.ndim == 3
                b_3d = b.ndim == 3

                # Extract shapes and batch sizes
                if a_3d:
                    batch_a, m, n = a.shape
                else:
                    m, n = a.shape
                    batch_a = 1

                if b_3d:
                    batch_b, n_b, p = b.shape
                else:
                    n_b, p = b.shape
                    batch_b = 1

                # Check for dimension compatibility
                assert_true(n == n_b, "Inner dimensions do not match for matrix multiplication")
                assert (
                    batch_a == batch_b or batch_a == 1 or batch_b == 1
                ), "Batch sizes must be equal or one must be 1"

                # Determine the batch size for the operation
                batch_size = batch_a
                c = zeros(shape=(batch_size, m, p))

                # Perform batched matrix multiplication
                for i in range(batch_size):
                    # Slice the batch or use the whole array if not batched
                    a_slice = a[i] if a_3d else a
                    b_slice = b[i] if b_3d else b

                    # Reshape for element-wise multiplication
                    a_reshaped = a_slice.reshape((m, n, 1))
                    b_reshaped = b_slice.reshape((1, n, p))

                    # Perform encrypted multiplication and sum along the axis
                    enc_mul_result = enc_mul(a_reshaped, b_reshaped)
                    c[i] = numpy.sum(enc_mul_result, axis=1)

                # Squeeze the first dimension if both inputs were 2D
                if not a_3d and not b_3d:
                    c = numpy.squeeze(c, axis=0)

                # Return the result of matrix multiplication
                return c

        # Remove the manual matrix multiplication when we can handle input precision with rounding
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4127
        @univariate
        def copy_function(x):
            return x

        # Copy trick explanation:
        # The copy_function is used to preserve the original precision of input values across
        # various operations. Operations like addition, subtraction, sum, and matmul can
        # unintentionally increase precision ('precision raising').
        #
        # Precision raising in one of these operations can inadvertently affect the precision of
        # the same value in other branches of the code. By creating copies of the input values,
        # any precision changes are limited to these copies, not the original values.
        #
        # This strategy is particularly important to make sure PBS in all branches are done on the
        # pre-defined precision. The use of the copy is conditional, applied only when needed
        # to optimize performance.

        input1_q_values_copy = (
            copy_function(input1_q_values) if is_encrypted_gemm else input1_q_values
        )
        input2_q_values_copy = (
            copy_function(input2_q_values) if is_encrypted_gemm else input2_q_values
        )

        # Core matmul operation in full integers with a shape change (INTEGERS)
        with tag(self.op_instance_name + ".matmul"):
            # We implement our own encrypted matmul to be able to round before PBS
            if is_encrypted_gemm:
                matmul = matmul(input1_q_values_copy, input2_q_values_copy)
            # Otherwise we let concrete do it
            else:
                matmul = input1_q_values_copy @ input2_q_values_copy

        input1_q_values_copy = (
            copy_function(input1_q_values) if is_encrypted_gemm else input1_q_values
        )

        # If the weights have symmetric quantization, their zero point will be 0
        # The following check avoids the computation of the sum of the inputs, which may have
        # large bit-width, in the case where it would be multiplied by zero
        if q_input2.quantizer.zero_point != 0:
            # Sum operation in full integers resulting in large integers (INTEGERS)
            with tag(self.op_instance_name + ".matmul_inputsum"):
                sum_input = -q_input2.quantizer.zero_point * numpy.sum(
                    input1_q_values_copy, axis=-1, keepdims=True
                )

            with tag(self.op_instance_name + ".matmul_add_inputsum"):
                # Last part that has to be done in integer
                numpy_q_out = matmul + sum_input
        else:
            numpy_q_out = matmul

        if self.debug_value_tracker is not None:
            # pylint: disable-next=unsubscriptable-object
            self.debug_value_tracker[self.op_instance_name]["output"] = numpy_q_out  # type: ignore

        input2_q_values_copy = (
            copy_function(input2_q_values) if is_encrypted_gemm else input2_q_values
        )

        with tag(self.op_instance_name + ".sum_weights_times_zero_point"):

            # sum_weights is a constant
            sum_weights = q_input1.quantizer.zero_point * numpy.sum(
                input2_q_values_copy, axis=-2, keepdims=True
            )

        final_term = p * q_input1.quantizer.zero_point * q_input2.quantizer.zero_point

        # Note that here we do not rescale to the output_scale and we do not add a zero-point
        # Any following Gemm/MatMul/Conv layers will do the rescaling (during re-quantization)
        # by calling _prepare_inputs_with_constants(...quantize_real_values=True)
        m_matmul = q_input1.quantizer.scale * q_input2.quantizer.scale

        # If this operation's result are network outputs, return
        # directly the integer values and a appropriate quantization parameters that
        # allow direct in-the-clear de-quantization, including the bias
        if self.produces_graph_output and not is_encrypted_gemm:
            out_zp: Union[int, numpy.ndarray] = sum_weights - final_term
            if q_bias is not None:
                # Make mypy happy
                assert q_bias is not None
                # Reshape the biases to broadcast them to each neuron
                bias_out = q_bias.values if isinstance(q_bias, QuantizedArray) else q_bias
                out_zp = out_zp + bias_out / (-m_matmul)

            # We identify terms in the above equation to determine what
            # the scale/zero-point of the in-the-clear quantizer should be
            # to properly de-quantize numpy_q_out
            return self.make_output_quant_parameters(numpy_q_out, m_matmul, out_zp)

        # Integer biases are only supported for Brevitas QAT which sets is_precomputed_qat to true
        # These biases are produced by QuantizedBrevitasQuant ops
        if q_bias is not None and q_bias.quantizer.is_precomputed_qat:
            # Make sure the scale was correctly matching during training
            # The bias scale should be the same scale as the one of the weights * inputs
            assert q_bias.quantizer.scale is not None
            assert numpy.isclose(q_bias.quantizer.scale, m_matmul)
            numpy_q_out += q_bias.qvalues

        # If weights are not encrypted then we can round as the next
        # line is going to be done in a PBS
        if not is_encrypted_gemm:
            with tag(self.op_instance_name + ".matmul_rounding"):
                # Apply Concrete rounding (if relevant)
                numpy_q_out = self.cnp_round(
                    numpy_q_out, calibrate_rounding, rounding_operation_id="matmul"
                )

                # Force a PBS with astype float64
                numpy_q_out = numpy_q_out.astype(numpy.float64)

        # Quantization scales and zero points
        # This is done in a PBS if is_encrypted_gemm == False
        # (along with the following activation function)
        # Otherwise it is done in FHE
        numpy_q_out = numpy_q_out + final_term - sum_weights

        if is_encrypted_gemm:
            with tag(self.op_instance_name + ".matmul_rounding"):
                # Apply Concrete rounding (if relevant)
                numpy_q_out = self.cnp_round(
                    numpy_q_out, calibrate_rounding, rounding_operation_id="matmul"
                )

        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None and not q_bias.quantizer.is_precomputed_qat:
            # The bias is handled as a float and will be fused
            numpy_q_out = numpy_q_out + q_bias.values

        # Return the float values, so that Concrete can fuse any following float operations
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

    def calibrate_analytical_output(self, *inputs: QuantizedArray) -> UniformQuantizer:
        """Calibrate output quantization based on analytical formulas.

        Args:
            *inputs (QuantizedArray): quantized operation inputs. Quantized weights
                are stored in the op instance

        Returns:
            res (UniformQuantizer): the quantizer of the operation's output values
                that can be used to de-quantize these values.
        """

        q_input1 = inputs[0]
        assert isinstance(q_input1, QuantizedArray)
        q_input2 = inputs[1]
        assert isinstance(q_input2, QuantizedArray)

        # In the operation Y = alpha * A' * B' + beta * C, q_bias is used for
        # generalised matrix multiplication. q_bias is set to None for standard
        # matrix multiplication (beta == 0 or only two inputs)
        q_bias = None if len(inputs) == 2 or self.attrs["beta"] == 0 else inputs[2]
        assert isinstance(q_bias, (type(None), QuantizedArray))

        # Using snake case here to please the Python format, the original attrs don't have the '_'
        # Use default false so we also support MatMul impl, MatMul does not have these flags
        transpose_inputs2 = self.attrs.get("transB", False)

        p = q_input2.qvalues.shape[-2]

        assert q_input1.quantizer.scale is not None
        assert q_input2.quantizer.scale is not None
        m_matmul = q_input1.quantizer.scale * q_input2.quantizer.scale

        input2_q_values = (
            numpy.transpose(q_input2.qvalues) if transpose_inputs2 else q_input2.qvalues
        )

        # Compute the third term, the sum of the weights which is a constant
        sum_weights = q_input1.quantizer.zero_point * numpy.sum(
            input2_q_values, axis=-2, keepdims=True
        )

        assert q_input1.quantizer.zero_point is not None
        assert q_input2.quantizer.zero_point is not None
        final_term = p * q_input1.quantizer.zero_point * q_input2.quantizer.zero_point

        out_zp: Union[int, numpy.ndarray] = sum_weights - final_term
        if q_bias is not None:
            # Make mypy happy
            assert q_bias is not None
            # Reshape the biases to broadcast them to each neuron
            bias_out = q_bias.values if isinstance(q_bias, QuantizedArray) else q_bias
            out_zp = out_zp + bias_out / (-m_matmul)

        out_params = UniformQuantizationParameters(
            scale=m_matmul,
            zero_point=out_zp,
            offset=0,
        )

        return UniformQuantizer(self._get_output_quant_opts(), self.output_quant_stats, out_params)


class QuantizedMatMul(QuantizedGemm):
    """Quantized MatMul op."""

    _impl_for_op_named: str = "MatMul"


class QuantizedAdd(QuantizedMixingOp):
    """Quantized Addition operator.

    Can add either two variables (both encrypted) or a variable and a constant
    """

    _impl_for_op_named: str = "Add"
    b_sign: int = 1

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        # If operating over all raw inputs, just perform the op in the clear
        if all(isinstance(q_input, RawOpOutput) for q_input in q_inputs):
            prepared_inputs = self._prepare_inputs_with_constants(
                *q_inputs, calibrate=False, quantize_actual_values=False
            )
            return self.call_impl(*prepared_inputs, **attrs).view(RawOpOutput)

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

        # Dequantize
        input_0 = q_input_0.dequant()
        input_1 = q_input_1.dequant()

        # If this operator is the last one in the graph,
        # we rescale using the smallest scale to keep all information
        if self.produces_graph_output:
            common_scale = min(q_input_0.quantizer.scale, q_input_1.quantizer.scale)
        # Otherwise we use the output op quantization scale
        else:
            common_scale = self.output_quant_params.scale

        common_zero_point = 0
        offset = 0

        output_quant_params = UniformQuantizationParameters(
            scale=common_scale,
            zero_point=common_zero_point,
            offset=offset,
        )

        quantizer = UniformQuantizer(params=output_quant_params, no_clipping=True)

        # Re-quantize using the common quantization paramaters
        q_input_0_rescaled = quantizer.quant(input_0)
        q_input_1_rescaled = quantizer.quant(input_1)

        # The sum of quantized encrypted integer values
        # This sum has << max(in_bits0, in_bits1) + 1 >> bits
        # Moreover, the zero-point will be sum of input zero-points
        assert self.b_sign in [-1, 1]

        sum_q = q_input_0_rescaled + self.b_sign * q_input_1_rescaled

        if self.produces_graph_output:
            return self.make_output_quant_parameters(sum_q, common_scale, common_zero_point)

        # But we would like the output to have n_bits, so we de-quantize
        dequant_sum = quantizer.dequant(sum_q)

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

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        assert_true(len(q_inputs) == 1, "Identity does not work with multiple QuantizedArray")

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        self.output_quant_params = q_inputs[0].quantizer.quant_params
        return super().q_impl(*q_inputs, **attrs)


class QuantizedReshape(QuantizedOp):
    """Quantized Reshape op."""

    _impl_for_op_named: str = "Reshape"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Reshape the input integer encrypted tensor.

        Args:
            q_inputs: an encrypted integer tensor at index 0 and one constant shape at index 1
            attrs: additional optional reshape options

        Returns:
            result (QuantizedArray): reshaped encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        newshape = prepared_inputs[1]
        assert_true(numpy.issubdtype(newshape.dtype, numpy.integer))

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, newshape, **attrs),
            value_is_float=False,
            options=prepared_inputs[0].quantizer.quant_options,
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Reshape operation can not be fused since it must be performed over integer tensors.

        Returns:
            bool: False, this operation can not be fused.
        """
        return False


class QuantizedConv(QuantizedMixingOp):
    """Quantized Conv op."""

    _impl_for_op_named: str = "Conv"

    def __init__(
        self,
        n_bits_output: int,
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        """Construct the quantized convolution operator and retrieve parameters.

        Args:
            n_bits_output: number of bits for the quantization of the outputs of this operator
            op_instance_name (str): The name that should be assigned to this operation, used
                to retrieve it later or get debugging information about this op (bit-width, value
                range, integer intermediary values, op-specific error messages). Usually this name
                is the same as the ONNX operation name for which this operation is constructed.
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

        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # Get the ONNX parameters
        self.group = attrs.get("group", 1)
        self.kernel_shape = attrs.get("kernel_shape", None)
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

        # Validate the parameters
        assert_true(
            len(self.kernel_shape) in (1, 2),
            "The convolution operator currently only supports 1d or 2d. "
            f"Got {len(self.kernel_shape)}-d",
        )
        assert_true(
            len(self.kernel_shape) == len(self.strides),
            "The convolution operator requires the number of strides to "
            "be the same as the number of kernel dimensions",
        )
        assert_true(
            bool(numpy.all(numpy.asarray(self.dilations) == 1)),
            "The convolution operator in Concrete does not support dilation",
        )
        assert_true(
            len(self.pads) == 2 * len(self.kernel_shape),
            "The convolution operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard",
        )

    # pylint: disable-next=too-many-statements, too-many-locals
    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Compute the quantized convolution between two quantized tensors.

        Allows an optional quantized bias.

        Args:
            q_inputs: input tuple, contains
                x (numpy.ndarray): input data. Shape is N x C x H x W for 2d
                w (numpy.ndarray): weights tensor. Shape is (O x I x Kh x Kw) for 2d
                b (numpy.ndarray, Optional): bias tensor, Shape is (O,)
            calibrate_rounding (bool): Whether to calibrate rounding
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

        assert q_weights.quantizer.scale is not None
        assert q_weights.quantizer.zero_point is not None

        assert q_input.quantizer.scale is not None
        assert q_input.quantizer.zero_point is not None

        # Can only pad with scalar zero-points, but zero-points can be float in special cases
        # for output layers
        _check_op_input_zero_point(q_input.quantizer.zero_point, self.op_instance_name)
        pad_value = int(q_input.quantizer.zero_point)
        q_input_pad = numpy_onnx_pad(q_input.qvalues, self.pads, pad_value, True)

        is_conv1d = len(self.kernel_shape) == 1

        q_weights_values = q_weights.qvalues
        kernel_shape = self.kernel_shape
        strides = self.strides
        dilations = self.dilations

        # Workaround for handling torch's Conv1d operator until it is supported by Concrete Python
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4117
        if is_conv1d:
            q_input_pad = numpy.expand_dims(q_input_pad, axis=-2)
            q_weights_values = numpy.expand_dims(q_weights_values, axis=-2)
            kernel_shape = (1, kernel_shape[0])
            strides = (1, strides[0])
            dilations = (1, dilations[0])

        # Prepare a constant tensor to compute the sum of the inputs
        q_weights_1 = numpy.ones_like(q_weights_values)

        # We follow the Quantized Gemm implementation
        # which in turn follows Eq.7 in https://arxiv.org/abs/1712.05877
        # to split the core computation from the zero points and scales.

        # Compute the first encrypted term that convolves weights and inputs
        # Force padding to 0 as padding needs to use a custom padding initializer
        # and is thus manually performed in the code above
        fake_pads = [0, 0] * len(kernel_shape)

        with tag(self.op_instance_name + ".conv"):
            conv_wx = fhe_conv(
                q_input_pad,
                q_weights_values,
                bias=None,
                pads=fake_pads,
                kernel_shape=kernel_shape,
                strides=strides,
                dilations=dilations,
                group=self.group,
            )

        # The total number of elements that are convolved by the application of a single kernel
        n_weights = numpy.prod(q_weights_values.shape[1:])

        # If the weights have symmetric quantization, their zero point will be 0
        # The following check avoids the computation of the sum of the inputs, which may have
        # large bit-width, in the case where it would be multiplied by zero
        if q_weights.quantizer.zero_point != 0:
            # Compute the sum of the inputs (second encrypted term)
            assert_true(
                isinstance(q_weights.quantizer.zero_point, (int, numpy.int_)),
                f"Zero point of weights tensor in {self.op_type} "
                f"op {self.op_instance_name} must be integer",
            )
            with tag(self.op_instance_name + ".conv_inputsum"):
                zw_conv_1x = -q_weights.quantizer.zero_point * fhe_conv(
                    q_input_pad,
                    q_weights_1,
                    bias=None,
                    pads=fake_pads,
                    kernel_shape=kernel_shape,
                    strides=strides,
                    dilations=dilations,
                    group=self.group,
                )

            with tag(self.op_instance_name + ".conv_add_inputsum"):
                numpy_q_out = conv_wx + zw_conv_1x
        else:
            numpy_q_out = conv_wx

        # Workaround for handling torch's Conv1d operator until it is supported by Concrete Python
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4117
        if is_conv1d:
            numpy_q_out = numpy.squeeze(numpy_q_out, axis=-2)

        if self.debug_value_tracker is not None:
            # pylint: disable-next=unsubscriptable-object
            self.debug_value_tracker[self.op_instance_name]["output"] = numpy_q_out

        weight_sum_axes = (1, 2) if is_conv1d else (1, 2, 3)
        weight_transpose_axes = (1, 0, 2) if is_conv1d else (1, 0, 2, 3)

        # Compute the third term, the sum of the weights which is a constant
        sum_weights = q_input.quantizer.zero_point * numpy.sum(
            q_weights.qvalues, axis=weight_sum_axes, keepdims=True
        ).transpose(*weight_transpose_axes)

        # Compute the forth term which is a constant
        final_term = n_weights * q_input.quantizer.zero_point * q_weights.quantizer.zero_point

        # Compute the rescaling factor that de-quantizes the input
        # This is going to be compiled with a PBS (along with the following activation function)
        # Note that we don't re-quantize the output of the conv, this will be done by
        # any Gemm/Add/Conv layers that follow
        m_matmul = q_input.quantizer.scale * q_weights.quantizer.scale

        bias_shape = (1, -1, 1) if is_conv1d else (1, -1, 1, 1)

        # If this operation's result are network outputs, return
        # directly the integer values and an appropriate quantization parameters that
        # allow direct in-the-clear de-quantization, including the bias
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
                out_zp = out_zp - q_bias.values.reshape(bias_shape) / m_matmul

            # We identify terms in the above equation to determine what
            # the scale/zero-point of the in-the-clear quantizer should be
            # to properly de-quantize numpy_q_out
            return self.make_output_quant_parameters(numpy_q_out, m_matmul, out_zp)

        if q_bias is not None and q_bias.quantizer.is_precomputed_qat:
            # Make sure the scale was correctly matching during training
            # The bias scale should be the same scale as the one of the weights * inputs
            assert q_bias.quantizer.scale is not None
            assert numpy.isclose(q_bias.quantizer.scale, m_matmul)

            numpy_q_out += q_bias.qvalues.reshape(bias_shape)

        with tag(self.op_instance_name + ".conv_rounding"):
            # Apply Concrete rounding (if relevant)
            numpy_q_out = self.cnp_round(
                numpy_q_out, calibrate_rounding, rounding_operation_id="matmul"
            )

        # Now compute the whole sum (sum of the four terms)
        numpy_q_out = numpy_q_out.astype(numpy.float64) + final_term - sum_weights

        # Rescale from scale=scale_inputs x scale_outputs to output scale
        numpy_q_out = m_matmul * numpy_q_out

        if q_bias is not None and not q_bias.quantizer.is_precomputed_qat:
            # The bias addition is handled in float and will be fused into a TLU
            # Reshape the biases to broadcast them to each channel
            numpy_q_out = numpy_q_out + q_bias.values.reshape(bias_shape)  # bias_part

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:

        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # Get the ONNX parameters
        self.ceil_mode = attrs.get("ceil_mode", 0)
        self.auto_pad = attrs.get("auto_pad", "NOTSET")
        self.kernel_shape = attrs.get("kernel_shape", None)

        assert_true(self.kernel_shape is not None, "Setting parameter 'kernel_shape' is required.")

        self.count_include_pad = attrs.get("count_include_pad", 1)
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

        # Validate the parameters
        assert_true(
            self.auto_pad == "NOTSET",
            "The 'auto_pad' parameter is not supported. Please keep the the default 'NOTSET' value "
            "and provide explicit padding.",
        )

        assert_true(
            len(self.kernel_shape) == 2,
            "The Average Pool operator currently supports only 2d kernels.",
        )

        assert_true(
            self.count_include_pad == 1,
            "Pad pixels must be included when calculating values on the edges. Please set "
            "'count_include_pad' to 1.",
        )

        assert_true(
            len(self.kernel_shape) == len(self.strides),
            "The Average Pool operator requires the number of strides to be the same as the number "
            "of kernel dimensions.",
        )

        assert_true(
            len(self.pads) == 2 * len(self.kernel_shape),
            "The Average Pool operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard.",
        )

        self.kernel: Union[numpy.ndarray, None] = None
        self.norm_const: Union[float, None] = None

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,
        **attrs,
    ) -> ONNXOpInputOutputType:

        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]

        n_in_channels = q_input.qvalues.shape[1]
        kernel = numpy.zeros(
            (n_in_channels, n_in_channels, self.kernel_shape[0], self.kernel_shape[1]),
            dtype=numpy.int64,
        )
        for i in range(n_in_channels):
            kernel[i, i, ::] = 1

        norm_const = 1.0 / onnx_avgpool_compute_norm_const(
            q_input.qvalues.shape,
            self.kernel_shape,
            self.pads,
            self.strides,
            self.ceil_mode,
        )

        # for mypy: The Quantized ops can only run on QuantizedArray that have quantization
        # parameters (i.e., were fully constructed). This should always be the case, except
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
            # Padding for TensorFlow style

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

        # Remark that here, we are _not_ using Concrete pad, since it would pad with
        # 0's while we want to pad with zero-point's. So, instead, he have done the padding
        # on our side, with q_input_pad
        fake_pads = [0] * len(self.pads)
        with tag(self.op_instance_name + ".avgpool"):
            sum_result = fhe_conv(q_input_pad, kernel, None, fake_pads, self.strides)

        with tag(self.op_instance_name + ".avgpool_rounding"):
            # Apply Concrete rounding (if relevant)
            sum_result = self.cnp_round(sum_result, calibrate_rounding)

        if self.debug_value_tracker is not None:
            # pylint: disable-next=unsubscriptable-object
            self.debug_value_tracker[self.op_instance_name]["output"] = sum_result

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:

        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # Get the ONNX parameters
        self.auto_pad = attrs.get("auto_pad", "NOTSET")
        self.ceil_mode = attrs.get("ceil_mode", 0)
        self.kernel_shape = attrs.get("kernel_shape", None)
        self.storage_order = attrs.get("storage_order", 0)
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

        # Validate the parameters
        assert_true(self.ceil_mode == 0, "Only ceil_mode = 0 is supported by Concrete for now")

        # Validate the parameters
        assert_true(
            len(self.kernel_shape) == 2,
            "The Max Pool operator currently supports only 2d",
        )
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
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:

        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]

        # for mypy: The Quantized ops can only run on QuantizedArray that have quantization
        # parameters (i.e., were fully constructed). This should always be the case, except
        # during the UniformQuantizer initialization when the zero_point can exist as None
        assert q_input.quantizer.zero_point is not None

        assert_true(
            self.ceil_mode == 0,
            "Only ceil_mode = 0 is supported by Concrete for now",
        )

        # Simple padding for PyTorch style
        pool_pads = compute_onnx_pool_padding(
            q_input.qvalues.shape,
            self.kernel_shape,
            self.pads,
            self.strides,
            self.ceil_mode,
        )

        q_input_pad = numpy_onnx_pad(
            q_input.qvalues, pool_pads, q_input.quantizer.zero_point, int_only=True
        )

        # Remark that here, we are _not_ using Concrete pad, since it would pad with
        # 0's while we want to pad with zero-point's. So, instead, he have done the padding
        # on our side, with q_input_pad
        fake_pads = [0] * len(self.pads)
        sum_result = fhe_maxpool(
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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:

        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # Get the ONNX parameters
        self.mode = attrs.get("mode", None)
        assert_true(
            self.mode == "constant",
            "Padding operator only supports padding with a constant",
        )

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,  # pylint: disable=unused-argument
        **attrs,
    ) -> ONNXOpInputOutputType:
        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]
        pads = prepared_inputs[1]

        assert_true(
            all(pads[i] == 0 and pads[4 + i] == 0 for i in range(0, 2)),
            "Concrete ML only supports padding along the width & height dimensions, padding"
            f" requested was {pads}",
        )

        assert_true(pads.size % 2 == 0)  # must be a multiple of 2
        assert_true(pads.size in [4, 8], "Only supporting pads of size 4 or 8.")
        if pads.size == 8:
            pads = numpy.asarray([pads[2], pads[3], pads[6], pads[7]])

        assert_true(pads.size == 4, "Not currently supporting padding of 3D tensors")

        pad_value = 0 if prepared_inputs[2] is None else prepared_inputs[2]
        assert_true(pad_value == 0, "Concrete ML only supports padding with constant zero values")

        assert q_input.quantizer.zero_point is not None
        q_input_pad = numpy_onnx_pad(q_input.qvalues, pads, q_input.quantizer.zero_point, True)

        return QuantizedArray(
            q_input.quantizer.n_bits,
            q_input_pad,
            value_is_float=False,
            options=q_input.quantizer.quant_options,
            stats=q_input.quantizer.quant_stats,
            params=q_input.quantizer.quant_params,
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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # We do not support testing a <= b where a,b are encrypted
        # only comparing to a constant is supported
        assert_true(constant_inputs is not None and len(constant_inputs) >= 1)


class QuantizedOr(QuantizedOpUnivariateOfEncrypted, QuantizedOp):
    """Or operator ||.

    This operation is not really working as a quantized operation. It just works when things got
    fused, as in e.g., Act(x) = x || (x + 42))
    """

    _impl_for_op_named: str = "Or"


class QuantizedDiv(QuantizedMixingOp):
    """Quantized Division operator.

    Can divide either two variables (both encrypted) or a variable and a constant
    """

    _impl_for_op_named: str = "Div"

    def __init__(
        self,
        *args,
        rounding_threshold_bits: Union[None, int, Dict[str, Union[str, int]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, rounding_threshold_bits=rounding_threshold_bits, **kwargs)
        self.divider_quantizer: Optional[UniformQuantizer] = None
        self.min_non_zero_value: Optional[numpy.float64] = None

    def calibrate(self, *inputs: Union[QuantizedArray, numpy.ndarray]) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            *inputs (Union[QuantizedArray, numpy.ndarray]): Calibration sample inputs.

        Returns:
            numpy.ndarray: the output values for the provided calibration samples.
        """

        # If the op can not be fused and that the two inputs are not constants
        # we need to compute the quantizer of the divider since we are doing
        # an encrypted division where both numerator and denominator are encrypted
        if not self.can_fuse() and len(inputs) == 2:
            assert isinstance(
                inputs[0], numpy.ndarray
            ), "Div calibrate does not support analytical calibration for now"
            assert isinstance(inputs[1], numpy.ndarray)

            # FIXME https://github.com/zama-ai/concrete-ml-internal/issues/4556
            min_non_zero_index = numpy.abs(inputs[1]).argmin(axis=None)
            min_non_zero_value = inputs[1].flat[min_non_zero_index]

            # mypy
            assert min_non_zero_value is not None and min_non_zero_value != 0
            self.min_non_zero_value = min_non_zero_value

            q_array_divider = QuantizedArray(self.n_bits, 1 / inputs[1])

            # Store the quantizer of the divider
            self.divider_quantizer = q_array_divider.quantizer

        return super().calibrate(*inputs)

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,
        **attrs,
    ) -> ONNXOpInputOutputType:

        # If the op can be fused we perform the op in the clear
        if self.can_fuse():
            return super().q_impl(*q_inputs, **attrs)

        # For mypy
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input_0: QuantizedArray = prepared_inputs[0]
        q_input_1: QuantizedArray = prepared_inputs[1]

        assert q_input_0.quantizer.scale is not None
        assert q_input_0.quantizer.zero_point is not None

        assert q_input_1.quantizer.scale is not None
        assert q_input_1.quantizer.zero_point is not None

        # Dequantize
        input_1 = q_input_1.dequant()

        # Replace input_1 with min_non_zero_value if input_1 is 0
        # mypy
        assert self.min_non_zero_value is not None
        input_1 = numpy.where(input_1 == 0, self.min_non_zero_value, input_1)

        # Compute the inverse of input_1
        input_1_inv = 1.0 / input_1

        # Re-quantize the inverse using the same quantization parameters as q_input_1
        # mypy
        assert self.divider_quantizer is not None
        # FIXME https://github.com/zama-ai/concrete-ml-internal/issues/4556
        q_input_1_inv_rescaled = self.divider_quantizer.quant(input_1_inv)

        # The product of quantized encrypted integer values
        product_q_values = q_input_0.qvalues * q_input_1_inv_rescaled

        # mypy
        assert q_input_0.quantizer.zero_point is not None
        assert q_input_1.quantizer.zero_point is not None
        assert self.divider_quantizer.zero_point is not None

        # Integer quantized multiplication need adjustment based on the zero points.
        if q_input_0.quantizer.zero_point:
            product_q_values -= q_input_0.quantizer.zero_point * (
                q_input_1_inv_rescaled - self.divider_quantizer.zero_point
            )
        if self.divider_quantizer.zero_point:
            product_q_values -= self.divider_quantizer.zero_point * (
                q_input_0.qvalues - q_input_0.quantizer.zero_point
            )

        # mypy
        assert self.divider_quantizer.scale is not None
        assert self.divider_quantizer.zero_point is not None

        # Compute the scale and zero point based on the scale and zero point
        # of the two quantized values multiplied together
        new_scale = q_input_0.quantizer.scale * self.divider_quantizer.scale
        new_zero_point = q_input_0.quantizer.zero_point * self.divider_quantizer.zero_point

        if self.produces_graph_output:
            return self.make_output_quant_parameters(product_q_values, new_scale, new_zero_point)

        with tag(self.op_instance_name + ".rounding"):
            # Apply Concrete rounding (if relevant)
            product_q_values = self.cnp_round(product_q_values, calibrate_rounding)

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4546
        # De-quantize the product
        dequant_product = (product_q_values - new_zero_point) * new_scale

        # Return the raw float values without re-quantizing them to the new scale, as any
        # following Gemm/Add/Conv will quantize them with _prepare_inputs_with_constants(...)
        return QuantizedArray(
            self.n_bits,
            dequant_product,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Div operation can be computed in float and fused if it operates over inputs produced
        by a single integer tensor.

        Returns:
            bool: Whether the number of integer input tensors allows computing this op as a TLU
        """

        return len(self._int_input_names) == 1


class QuantizedMul(QuantizedMixingOp):
    """Quantized Multiplication operator.

    Can multiply either two variables (both encrypted) or a variable and a constant
    """

    _impl_for_op_named: str = "Mul"

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,
        **attrs,
    ) -> ONNXOpInputOutputType:

        # If either input is a RawOpOutput or if the op can be fused,
        # perform the op in the TLU using FP32
        if (
            len(q_inputs) == 1
            or isinstance(q_inputs[0], RawOpOutput)
            or isinstance(q_inputs[1], RawOpOutput)
            or self.can_fuse()
        ):
            return super().q_impl(*q_inputs, **attrs)

        # For mypy
        assert self.output_quant_params is not None
        assert self.output_quant_params.scale is not None
        assert self.output_quant_params.zero_point is not None

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input_0: QuantizedArray = prepared_inputs[0]
        q_input_1: QuantizedArray = prepared_inputs[1]

        assert q_input_0.quantizer.scale is not None
        assert q_input_0.quantizer.zero_point is not None

        assert q_input_1.quantizer.scale is not None
        assert q_input_1.quantizer.zero_point is not None

        # Remove the manual encrypted multiplication when we
        # can handle input precision with rounding
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4127
        @univariate
        def copy_function(x):
            return x

        input0_q_values = q_input_0.qvalues
        input1_q_values = q_input_1.qvalues

        with tag(self.op_instance_name + ".enc_mul_rounding_input0"):
            input0_q_values = self.cnp_round(
                input0_q_values, calibrate_rounding, rounding_operation_id="input0_q_values_copy"
            )

        with tag(self.op_instance_name + ".enc_mul_rounding_input1"):
            input1_q_values = self.cnp_round(
                input1_q_values, calibrate_rounding, rounding_operation_id="input1_q_values_copy"
            )

        input0_q_values = copy_function(input0_q_values)
        input1_q_values = copy_function(input1_q_values)

        # The product of quantized encrypted integer values
        product_q_values = input0_q_values * input1_q_values

        # Integer quantized multiplication need adjustment based on the zero points.
        if q_input_0.quantizer.zero_point:
            product_q_values -= q_input_0.quantizer.zero_point * input1_q_values
        if q_input_1.quantizer.zero_point:
            product_q_values -= q_input_1.quantizer.zero_point * input0_q_values

        # Compute the scale and zero point based on the scale and zero point
        # of the two quantized values multiplied together
        new_scale = q_input_0.quantizer.scale * q_input_1.quantizer.scale
        new_zero_point = q_input_0.quantizer.zero_point * q_input_1.quantizer.zero_point

        if self.produces_graph_output:
            return self.make_output_quant_parameters(product_q_values, new_scale, new_zero_point)

        with tag(self.op_instance_name + ".rounding"):
            # Apply Concrete rounding (if relevant)
            product_q_values = self.cnp_round(
                product_q_values, calibrate_rounding, rounding_operation_id="product_q_values"
            )

        # De-quantize the product
        dequant_product = (product_q_values + new_zero_point) * new_scale

        # Return the raw float values without re-quantizing them to the new scale, as any
        # following Gemm/Add/Conv will quantize them with _prepare_inputs_with_constants(...)
        return QuantizedArray(
            self.n_bits,
            dequant_product,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Mul operation can be computed in float and fused if it operates over inputs produced
        by a single integer tensor.

        Returns:
            bool: Whether the number of integer input tensors allows computing this op as a TLU
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

    def calibrate(self, *inputs: Union[QuantizedArray, numpy.ndarray]) -> numpy.ndarray:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            *inputs (numpy.ndarray): Calibration sample inputs.

        Returns:
            numpy.ndarray: the output values for the provided calibration samples.
        """

        assert all(
            isinstance(inp, numpy.ndarray) for inp in inputs
        ), "Batch Normalization calibrate does not support analytical calibration"

        # Here we need the actual values of the constants, we need to pass through
        # the numpy.ndarrays in the computation graph
        prepared_inputs = self._prepare_inputs_with_constants(
            *inputs, calibrate=True, quantize_actual_values=False
        )

        raw_result = self.call_impl(*prepared_inputs, **self.attrs)

        if not isinstance(raw_result, RawOpOutput):
            # Check if batch normalization is applied per channel
            scale = self.constant_inputs[self._params_name_to_input_idx["scale"]].values
            bias = self.constant_inputs[self._params_name_to_input_idx["bias"]].values
            if scale.size > 1 or bias.size > 1:
                # Per channel batchnorm struggles with low bit-width quantization.
                # Batchnorm parameters (scale/bias/mean/var) can vary significantly
                # between channels. Our tensor-based quantization, however, only supports
                # a global scale and offset. Errors in quantized values can thus be severe.
                # To mitigate this, we use percentiles to clip extreme values.

                lower_bound = numpy.percentile(raw_result, 0.1)
                upper_bound = numpy.percentile(raw_result, 99.9)

                raw_result = numpy.clip(raw_result, lower_bound, upper_bound)

            quantized_samples = QuantizedArray(self.n_bits, raw_result)

            self.output_quant_params = quantized_samples.quantizer.quant_params
            self.output_quant_stats = quantized_samples.quantizer.quant_stats

            raw_result = quantized_samples.values

        return raw_result


class QuantizedFlatten(QuantizedOp):
    """Quantized flatten for encrypted inputs."""

    _impl_for_op_named: str = "Flatten"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Flatten the input integer encrypted tensor.

        Args:
            q_inputs: an encrypted integer tensor at index 0
            attrs: contains axis attribute

        Returns:
            result (QuantizedArray): reshaped encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

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
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        """Construct the quantized ReduceSum operator and retrieve parameters.

        Args:
            n_bits_output (int): Number of bits for the operator's quantization of outputs.
            op_instance_name (str): The name that should be assigned to this operation, used
                to retrieve it later or get debugging information about this op (bit-width, value
                range, integer intermediary values, op-specific error messages). Usually this name
                is the same as the ONNX operation name for which this operation is constructed.
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
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # Retrieve and set the ONNX parameters
        # Numpy's keepdims parameter is a boolean while ONNX's one is an int (0 or 1). Even though
        # Python handles them equivalently, we need to manually convert it as mypy doesn't accept
        # this type difference
        self.keepdims = bool(attrs.get("keepdims", 1))
        self.noop_with_empty_axes = attrs.get("noop_with_empty_axes", 0)
        self.copy_inputs = False

    def calibrate(self, *inputs: Union[QuantizedArray, numpy.ndarray]) -> numpy.ndarray:
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

        quantized_samples = QuantizedArray(
            self.n_bits, self.call_impl(*prepared_inputs, **self.attrs)
        )

        self.output_quant_params = quantized_samples.quantizer.quant_params
        self.output_quant_stats = quantized_samples.quantizer.quant_stats

        return quantized_samples.values

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        calibrate_rounding: bool = False,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Sum the encrypted tensor's values along the given axes.

        Args:
            q_inputs (QuantizedArray): An encrypted integer tensor at index 0.
            calibrate_rounding (bool): Whether to calibrate rounding or not.
            attrs (Dict): Options are handled in constructor.

        Returns:
            (QuantizedArray): The sum of all values along the given axes.
        """
        # Retrieve the quantized inputs as well as the function's constant parameters
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        assert_true(
            isinstance(prepared_inputs[0], (QuantizedArray)),
            "Prepared inputs' values should be found in a QuantizedArray. "
            f"Got {type(prepared_inputs[0])}",
        )

        assert_true(
            isinstance(prepared_inputs[1], numpy.ndarray),
            "ReduceSum axis parameter should be a Numpy array. " f"Got {type(prepared_inputs[1])}",
        )

        # Retrieve the quantized values and the axes to sum over
        # Parameter "axis" Numpy's sum operator has to be a tuple (and not an array)
        q_values = prepared_inputs[0].qvalues
        axes = tuple(prepared_inputs[1])

        assert_true(
            0 not in axes,
            "ReduceSum axis parameter should not contain axis 0 as it is used for batching "
            f"the inference. Got {axes}",
        )

        with tag(self.op_instance_name):

            # Need to copy to prevent the following sum to raise precision of the input
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4127
            @univariate
            def copy_function(x):
                return x

            if self.copy_inputs:
                q_values = copy_function(q_values)

            # Sum all the quantized values
            q_sum = numpy.sum(q_values, axis=axes, keepdims=self.keepdims)

            # Determining the number of output zero_points to use for de-quantization with
            # the total number of elements summed all together, which is the product of all
            # the number of elements found within the given axes
            n_elem = numpy.prod([q_values.shape[axis] for axis in axes])

            # Determining the output scale and zero_point
            input_quantizer = prepared_inputs[0].quantizer
            scale = input_quantizer.scale
            zero_point = n_elem * input_quantizer.zero_point

            # If this operator is the graph's last operator,
            # there's is no need to created additional TLUs
            if self.produces_graph_output:
                return self.make_output_quant_parameters(q_sum, scale, zero_point)

            f_substract = q_sum - zero_point

        with tag(self.op_instance_name + ".rounding"):
            # Apply Concrete rounding (if relevant)
            f_substract = self.cnp_round(f_substract, calibrate_rounding)

        # De-quantize the sum
        f_sum = scale * f_substract

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
    # Note that this should be reset when the correctness test that finds
    # all mismatches between Concrete ML and Brevitas is fixed
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2373
    quantize_inputs_with_model_outputs_precision = True
    output_quant_opts: QuantizationOptions

    def __init__(
        self,
        n_bits_output: int,
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        """Construct the Brevitas quantization operator.

        Args:
            n_bits_output (int): Number of bits for the operator's quantization of outputs.
                Not used, will be overridden by the bit_width in ONNX
            op_instance_name (str): The name that should be assigned to this operation, used
                to retrieve it later or get debugging information about this op (bit-width, value
                range, integer intermediary values, op-specific error messages). Usually this name
                is the same as the ONNX operation name for which this operation is constructed.
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
                    e.g., [-2**n_bits-1 .. 2**n_bits-1] (default 0),
        """

        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

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

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4544
        # Remove this workaround when brevitas export is fixed
        if self.is_signed is False and self.is_narrow is True:
            self.is_signed = True
            self.is_narrow = False

        assert_false(
            not self.is_signed and self.is_narrow,
            "Can not use narrow range for non-signed Brevitas quantizers",
        )

        # To ensure de-quantization produces floats, the following parameters must be float.
        # This should be default export setting in Brevitas
        check_float(
            self.constant_inputs is not None
            and self.constant_inputs[self._params_name_to_input_idx["scale"]],
            "Scale of Brevitas Quant op must be float",
        )
        check_float(
            self.constant_inputs is not None
            and self.constant_inputs[self._params_name_to_input_idx["zero_point"]],
            "Zero Point of Brevitas Quant op must be float",
        )

        # For mypy
        assert self.constant_inputs is not None

        # The constant inputs can have either int or str keys, here it is an int
        n_bits = self.constant_inputs[3]  # type: ignore

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

    def calibrate(self, *inputs: Union[QuantizedArray, numpy.ndarray]) -> numpy.ndarray:
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

        n_bits = int(self.constant_inputs[3])

        self.output_quant_params = UniformQuantizationParameters(
            scale=numpy.float64(self.constant_inputs[1]),
            zero_point=int(self.constant_inputs[2]),
            offset=2 ** (n_bits - 1) if self.is_signed else 0,
        )

        return result

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Quantize values.

        Args:
            q_inputs: an encrypted integer tensor at index 0,
                      scale, zero_point, n_bits at indices 1,2,3
            attrs: additional optional attributes

        Returns:
            result (QuantizedArray): reshaped encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=False
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        # Impose the number of bits that this op quantizes its inputs, to the value that
        # was provided by Brevitas. This number of bits is normally
        # different from the op's n_bits. The op will usually have an output number of bits
        # that is suitable for network output, as the op could be the last op in the graph

        # Thus, we enforce the QAT number of bits on this op's output
        n_bits = int(prepared_inputs[3])

        assert len(q_inputs) >= 1

        assert_true(
            self.output_quant_params is not None,
            "You need to calibrate this op before using it",
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

    This operator performs quantization and transposes the encrypted data.
    When the inputs are pre-computed QAT the input is only quantized if needed.
    """

    _impl_for_op_named: str = "Transpose"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Transpose the input integer encrypted tensor.

        Args:
            q_inputs: an encrypted integer tensor at index 0 and one constant shape at index 1
            attrs: additional optional reshape options

        Returns:
            result (QuantizedArray): transposed encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, **attrs),
            value_is_float=False,
            options=prepared_inputs[0].quantizer.quant_options,
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Transpose can not be fused since it must be performed over integer tensors as
        it moves around different elements of these input tensors.

        Returns:
            bool: False, this operation can not be fused as it copies encrypted integers
        """
        return False


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

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Unsqueeze the input tensors on a given axis.

        Args:
            q_inputs: an encrypted integer tensor at index 0, axes at index 1
            attrs: additional optional unsqueeze options

        Returns:
            result (QuantizedArray): unsqueezed encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        axes = prepared_inputs[1]

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, axes, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Unsqueeze can not be fused since it must be performed over integer tensors as
        it reshapes an encrypted tensor.

        Returns:
            bool: False, this operation can not be fused as it operates on encrypted tensors
        """
        return False


class QuantizedConcat(QuantizedOp):
    """Concatenate operator."""

    _impl_for_op_named: str = "Concat"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Concatenate the input tensors on a given axis.

        Args:
            q_inputs: an encrypted integer tensor
            attrs: additional optional concatenate options

        Returns:
            result (QuantizedArray): concatenated encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        # The input tensors must have the same quantization parameters to be concatenated.
        scales = [x.quantizer.scale for x in prepared_inputs]
        zero_points = [x.quantizer.zero_point for x in prepared_inputs]
        assert_true(
            all(x == scales[0] for x in scales) and all(x == zero_points[0] for x in zero_points),
            "All inputs must have the same scale and zero_point to be concatenated.",
        )
        assert_true(
            all(
                prep_input.quantizer.quant_options.is_equal(
                    prepared_inputs[0].quantizer.quant_options
                )
                for prep_input in prepared_inputs[1:]
            )
        )

        tensors_to_concat = [prepared_input.qvalues for prepared_input in prepared_inputs]

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(*tensors_to_concat, **attrs),
            value_is_float=False,
            options=prepared_inputs[0].quantizer.quant_options,
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Concatenation can not be fused since it must be performed over integer tensors as
        it copies encrypted integers from one tensor to another.

        Returns:
            bool: False, this operation can not be fused as it copies encrypted integers
        """
        return False


class QuantizedSqueeze(QuantizedOp):
    """Squeeze operator."""

    _impl_for_op_named: str = "Squeeze"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Squeeze the input tensors on a given axis.

        Args:
            q_inputs: an encrypted integer tensor at index 0, axes at index 1
            attrs: additional optional squeeze options

        Returns:
            result (QuantizedArray): squeezed encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        axes = prepared_inputs[1]

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, axes, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Squeeze can not be fused since it must be performed over integer tensors as
        it reshapes encrypted tensors.

        Returns:
            bool: False, this operation can not be fused as it reshapes encrypted tensors
        """
        return False


class ONNXShape(QuantizedOp):
    """Shape operator."""

    _impl_for_op_named: str = "Shape"

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        # This op takes only encrypted inputs in the form of QuantizedArray
        assert isinstance(q_inputs[0], QuantizedArray)

        return numpy.asarray(q_inputs[0].qvalues.shape, numpy.int64).view(RawOpOutput)

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        This operation returns the shape of the tensor and thus can not be fused into a
        univariate TLU.

        Returns:
            bool: False, this operation can not be fused
        """
        return False


class ONNXConstantOfShape(QuantizedOp):
    """ConstantOfShape operator."""

    _impl_for_op_named: str = "ConstantOfShape"

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        This operation returns a new encrypted tensor and thus can
        not be fused.

        Returns:
            bool: False, this operation can not be fused
        """
        return False


class ONNXGather(QuantizedOp):
    """Gather operator.

    Returns values at requested indices from the input tensor.
    """

    _impl_for_op_named: str = "Gather"

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # The first parameter can be either an encrypted tensor or a shape (int64 array)
        if isinstance(prepared_inputs[0], QuantizedArray):
            inputs = (prepared_inputs[0].qvalues, *prepared_inputs[1:])
            return QuantizedArray(
                prepared_inputs[0].quantizer.n_bits,
                self.call_impl(*inputs, **attrs),
                value_is_float=False,
                options=prepared_inputs[0].quantizer.quant_options,
                stats=prepared_inputs[0].quantizer.quant_stats,
                params=prepared_inputs[0].quantizer.quant_params,
            )

        assert_true(isinstance(prepared_inputs[0], numpy.ndarray))
        return self.call_impl(*prepared_inputs, **attrs)

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        This operation returns values from a tensor and thus can not be fused into a
        univariate TLU.

        Returns:
            bool: False, this operation can not be fused
        """
        return False


class ONNXSlice(QuantizedOp):
    """Slice operator."""

    _impl_for_op_named: str = "Slice"

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        assert_true(
            isinstance(prepared_inputs[0], QuantizedArray),
            "Slice currently only supports QuantizedArray inputs (encrypted inputs)",
        )

        inputs = (prepared_inputs[0].qvalues, *prepared_inputs[1:])
        return QuantizedArray(
            prepared_inputs[0].quantizer.n_bits,
            self.call_impl(*inputs, **attrs),
            value_is_float=False,
            options=prepared_inputs[0].quantizer.quant_options,
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        This operation returns values from a tensor and thus can not be fused into a
        univariate TLU.

        Returns:
            bool: False, this operation can not be fused
        """
        return False


class QuantizedExpand(QuantizedOp):
    """Expand operator for quantized tensors."""

    _impl_for_op_named: str = "Expand"
    quantize_inputs_with_model_outputs_precision = True

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Expand the input tensor to a specified shape.

        Args:
            q_inputs: an encrypted integer tensor at index 0, shape at index 1
            attrs: additional optional expand options

        Returns:
            result (QuantizedArray): expanded encrypted integer tensor
        """

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        # Ensure the input is a quantized array
        assert isinstance(q_inputs[0], QuantizedArray)

        target_shape = prepared_inputs[1]

        # Return a new quantized array with the same quantization parameters
        return QuantizedArray(
            q_inputs[0].quantizer.n_bits,
            self.call_impl(prepared_inputs[0].qvalues, target_shape, **attrs),
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=prepared_inputs[0].quantizer.quant_stats,
            params=prepared_inputs[0].quantizer.quant_params,
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Unsqueeze can not be fused since it must be performed over integer tensors as
        it reshapes an encrypted tensor.

        Returns:
            bool: False, this operation can not be fused as it operates on encrypted tensors
        """
        return False


class QuantizedEqual(QuantizedOp):
    """Comparison operator ==.

    Only supports comparison with a constant.
    """

    _impl_for_op_named: str = "Equal"

    # Since this op takes a single variable input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # We do not support testing a == b where a,b are encrypted
        # only comparing to a constant is supported
        assert_true(constant_inputs is not None and len(constant_inputs) >= 1)


class QuantizedUnfold(QuantizedMixingOp):
    """Quantized Unfold op."""

    _impl_for_op_named: str = "Unfold"

    # Since this op takes a single input, we can set int_input_names to a single default id
    def __init__(
        self,
        n_bits_output: int,
        op_instance_name: str,
        int_input_names: Set[str] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: QuantizationOptions = None,
        **attrs,
    ) -> None:

        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )

        # Get the ONNX parameters
        self.kernel_shape = attrs.get("kernel_shape", None)
        self.pads = attrs.get("pads", tuple([0] * 2 * (len(self.kernel_shape) - 2)))
        self.dilations = attrs.get("dilations", tuple([1] * len(self.kernel_shape)))
        self.strides = attrs.get("strides", tuple([1] * len(self.kernel_shape)))

        # Validate the parameters
        assert_true(
            len(self.kernel_shape) == 2,
            "The Unfold operator currently supports only 2d",
        )
        assert_true(
            len(self.kernel_shape) == len(self.strides),
            "The Unfold operator requires the number of strides to "
            "be the same as the number of kernel dimensions",
        )
        assert_true(
            len(self.pads) == 2 * len(self.kernel_shape),
            "The Unfold operator in Concrete ML requires padding to be specified as "
            " (pad_left_dim1, pad_right_dim1, pad_left_dim2, pad_right_dim2, ...), following ONNX"
            " standard",
        )

        self.kernel: Union[numpy.ndarray, None] = None
        self.norm_const: Union[float, None] = None

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:

        # Retrieve the quantized inputs
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )
        q_input: QuantizedArray = prepared_inputs[0]

        n_in_channels = q_input.qvalues.shape[1]
        kernels_list = []
        for _ in range(n_in_channels):
            for row in range(self.kernel_shape[0]):
                for col in range(self.kernel_shape[1]):
                    kernel = numpy.zeros(
                        (1, 1, self.kernel_shape[0], self.kernel_shape[1]),
                        dtype=numpy.int64,
                    )
                    kernel[:, :, row, col] = 1
                    kernels_list.append(kernel)
        kernels = numpy.concatenate(numpy.array(kernels_list), axis=0)

        # for mypy: The Quantized ops can only run on QuantizedArray that have quantization
        # parameters (i.e., were fully constructed). This should always be the case, except
        # during the UniformQuantizer initialization when the zero_point can exist as None
        assert q_input.quantizer.zero_point is not None

        # Compute padding with floor and apply it to the input, pad with the input zero-point
        pool_pads = compute_onnx_pool_padding(
            q_input.qvalues.shape, self.kernel_shape, self.pads, self.strides, ceil_mode=0
        )

        # Can only pad with scalar zero-points, but zero-points can be float in special cases
        # for output layers
        _check_op_input_zero_point(q_input.quantizer.zero_point, self.op_instance_name)
        pad_value = int(q_input.quantizer.zero_point)
        q_input_pad = numpy_onnx_pad(q_input.qvalues, pool_pads, pad_value, int_only=True)

        # Remark that here, we are _not_ using Concrete pad, since it would pad with
        # 0's while we want to pad with zero-point's. So, instead, he have done the padding
        # on our side, with q_input_pad
        fake_pads = [0] * len(self.pads)

        with tag(self.op_instance_name + ".unfold"):
            sum_result = fhe_conv(
                q_input_pad, kernels, None, fake_pads, self.strides, None, None, n_in_channels
            )

        if self.debug_value_tracker is not None:
            # pylint: disable-next=unsubscriptable-object
            self.debug_value_tracker[self.op_instance_name][
                "output"
            ] = sum_result  # pragma: no cover

        result = (
            sum_result.astype(numpy.float64) - q_input.quantizer.zero_point
        ) * q_input.quantizer.scale

        # Reshape to fit the same shape output as unfold
        result = result.reshape((result.shape[0], result.shape[1], -1))

        return QuantizedArray(
            self.n_bits,
            result,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )
