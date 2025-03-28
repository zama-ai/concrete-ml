"""GLWE backend for some supported layers."""

import json

import numpy
import torch

from ..common.utils import HybridFHEMode, to_tuple
from ..quantization.quantized_ops import QuantizedGemm
from .quantized_module import QuantizedModule
from .quantizers import TorchUniformQuantizer


def has_glwe_backend():
    """Check if the GLWE backend is installed.

    Returns:
        bool: True if the GLWE backend is installed, False otherwise.
    """
    try:
        __import__("concrete_ml_extensions")
        return True
    except ImportError:  # pragma: no cover
        return False  # pragma: no cover


class GLWELinearLayerExecutor:
    """GLWE execution helper for pure linear layers."""

    def __init__(
        self,
        private_key=None,
        compression_key=None,
        use_dynamic_quantization: bool = True,
    ):
        """Initialize the GLWE linear layer executor.

        Args:
            private_key: Private key for encryption
            compression_key: Compression key for encryption
            use_dynamic_quantization: Whether to use dynamic quantization
        """
        assert has_glwe_backend(), "GLWE backend not installed"

        import concrete_ml_extensions as fhext

        self.fhext = fhext

        self.compression_key = compression_key
        self.private_key = private_key

        default_crypto_params_glwe = json.loads(fhext.default_params())  # pylint: disable=no-member
        self.glwe_crypto_params = (
            fhext.MatmulCryptoParameters.deserialize(  # pylint: disable=no-member
                json.dumps(default_crypto_params_glwe)
            )
        )
        self.poly_size = default_crypto_params_glwe["packing_ks_polynomial_size"]
        self.calibrated_max_bits = default_crypto_params_glwe["bits_reserved_for_computation"]
        self.use_dynamic_quantization = use_dynamic_quantization

    def keygen(self):
        """Generate private and compression key."""
        # pylint: disable-next=no-member
        self.private_key, self.compression_key = self.fhext.create_private_key(
            self.glwe_crypto_params
        )

    def _get_quant_range(self, q_module: QuantizedModule, is_signed: bool = False):
        """Return the minimum and maximum signed quantized values for the given module.

        Args:
            q_module: The quantized module to get the range for
            is_signed: Whether to return the signed range

        Returns:
            tuple: Minimum and maximum quantized values
        """
        input_n_bits = q_module.input_quantizers[0].quant_options.n_bits
        if is_signed:
            return -(2 ** (input_n_bits - 1)), 2 ** (input_n_bits - 1) - 1
        return 0, 2**input_n_bits - 1

    def _per_channel_weight_quantization(
        self, weight: numpy.ndarray, q_module: QuantizedModule, device: torch.device
    ):
        """Quantize the weights, per-channel using symmetric (signed) quantization.

        Args:
            weight: Weight tensor to quantize
            q_module: Quantized module containing quantization parameters
            device: Device to place tensors on

        Returns:
            tuple: Quantized weights, scale, zero point and weight sum
        """
        weight_float = torch.from_numpy(weight).to(device)

        # Get the signed integer range
        q_min, q_max = self._get_quant_range(q_module, is_signed=True)

        w_min_vals, _ = weight_float.min(dim=0, keepdim=True)
        w_max_vals, _ = weight_float.max(dim=0, keepdim=True)

        weight_scale = (w_max_vals - w_min_vals) / (q_max - q_min)
        # Avoid division by zero.
        weight_scale = torch.where(
            (w_max_vals > w_min_vals), weight_scale, torch.ones_like(weight_scale)
        )
        weight_scale = weight_scale.squeeze(-1)  # shape: (out_dim,)

        # Quantization
        weight_zp = torch.round(q_min - w_min_vals / weight_scale).to(torch.float32)

        # Apply quantization with proper broadcasting
        weight_q = torch.round(weight_float / weight_scale) + weight_zp
        weight_q = torch.clamp(weight_q, q_min, q_max).to(torch.float32)
        sum_w = weight_q.sum(dim=0)  # sum over the input dimension

        return weight_q, weight_scale, weight_zp, sum_w

    def _dynamic_input_quantization(
        self, x: torch.Tensor, q_module: QuantizedModule, transpose_inputs: bool = False
    ):
        """Dynamically quantize the input tensor on a per-sample basis.

        Args:
            x: Input tensor to quantize
            q_module: Quantized module containing quantization parameters
            transpose_inputs: Whether to transpose inputs

        Returns:
            tuple: Quantized input, scale, zero point and original shape
        """
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.view(-1, original_shape[-1])
        else:
            x_flat = x

        q_min, q_max = self._get_quant_range(q_module)

        rmin = x_flat.min(dim=1, keepdim=True).values
        rmax = x_flat.max(dim=1, keepdim=True).values

        x_scale = (rmax - rmin) / (q_max - q_min)
        x_scale = torch.where(rmax > rmin, x_scale, torch.ones_like(x_scale))
        x_zp = torch.round((q_min - rmin) / x_scale).to(torch.float32)

        x_q = torch.round(x_flat / x_scale) + x_zp
        x_q = torch.clamp(x_q, q_min, q_max).to(torch.float32)

        x_q = x_q.transpose(-1, -2) if transpose_inputs else x_q

        return x_q, x_scale, x_zp, original_shape

    def _apply_correction_and_dequantize(
        self,
        raw: torch.Tensor,
        x_q: torch.Tensor,
        x_zp: torch.Tensor,
        weight_zp: torch.Tensor,
        sum_w: torch.Tensor,
        k: int,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Apply zero-point correction and de-quantize the result.

        Args:
            raw: Raw matrix multiplication result
            x_q: Quantized input
            x_zp: Input zero point
            weight_zp: Weight zero point
            sum_w: Sum of weights
            k: Input dimension
            x_scale: Input scale
            weight_scale: Weight scale

        Returns:
            torch.Tensor: Dequantized result
        """
        # Compute sum of quantized input values.
        sum_x = x_q.sum(dim=-1, keepdim=True).long()

        assert raw.dim() == 2 or raw.dim() == 3, "Unsupported raw tensor dimension."

        # Broadcast weight quantization parameters.
        if raw.dim() == 2:
            # raw shape: (N, out_dim)
            weight_zp_broadcast = weight_zp.view(1, -1)
            sum_w_broadcast = sum_w.view(1, -1)
        else:
            # raw shape: (batch, n_rows, out_dim)
            weight_zp_broadcast = weight_zp.view(1, 1, -1)
            sum_w_broadcast = sum_w.view(1, 1, -1)

        # Apply correction:
        #   raw - [weight_zp * sum_x + x_zp * sum_w - x_zp * weight_zp * k]
        correction = (
            (weight_zp_broadcast * sum_x)
            + (x_zp * sum_w_broadcast)
            - (x_zp * weight_zp_broadcast * k)
        )
        acc = raw - correction

        # Dequantize
        if raw.dim() == 2:
            scale_product = x_scale * weight_scale.view(1, -1)
        else:  # raw.dim() == 3
            scale_product = x_scale * weight_scale.view(1, 1, -1)
        return acc.float() * scale_product

    def _extract_weight_params(self, quantized_linear_op, transpose_inputs2: bool):
        """Extract and possibly transpose weights.

        Args:
            quantized_linear_op: The quantized linear operation
            transpose_inputs2: Whether to transpose inputs

        Returns:
            tuple: Full precision and quantized weights
        """
        # Get the weight constant (assumed to be the first constant input).
        weight_arr = list(quantized_linear_op.constant_inputs.values())[0]
        # Ensure symmetric quantization.
        assert weight_arr.quantizer.quant_params.zero_point == 0

        weight = weight_arr.values
        qweight = weight_arr.qvalues.astype(numpy.float32)

        if transpose_inputs2:
            weight = numpy.transpose(weight)
            qweight = numpy.transpose(qweight)
        return weight, qweight

    def _add_bias(
        self, out_tensor: torch.Tensor, quantized_layer, device: torch.device
    ) -> torch.Tensor:
        """Add bias to the output tensor if present.

        Args:
            out_tensor: The tensor to add bias to
            quantized_layer: The quantized layer containing bias information
            device: The device to place the bias tensor on

        Returns:
            torch.Tensor: Tensor with bias added
        """
        if len(quantized_layer[1].constant_inputs) > 1:
            bias = list(quantized_layer[1].constant_inputs.values())[1].values
            bias = torch.from_numpy(bias).to(device)
            out_tensor += bias
        return out_tensor

    def _forward_clear(
        self,
        x: torch.Tensor,
        q_module: QuantizedModule,
        transpose_inputs1: bool,
        qweight: numpy.ndarray,
        weight: numpy.ndarray,
    ) -> torch.Tensor:
        """Forward pass in clear (non-encrypted) mode.

        Args:
            x: Input tensor
            q_module: Quantized module containing quantization parameters
            transpose_inputs1: Whether to transpose the first input
            qweight: Quantized weight matrix
            weight: Original weight matrix in full precision (required for dynamic quantization)

        Returns:
            torch.Tensor: Output tensor after applying the linear operation
        """
        if not self.use_dynamic_quantization:
            # Static quantization implementation.
            quantizer = TorchUniformQuantizer(q_module.input_quantizers[0])
            out_quantizer = TorchUniformQuantizer(q_module.output_quantizers[0])

            q_x_torch = quantizer.quant(x, dtype=torch.float32)
            q_x_torch = torch.transpose(q_x_torch, 1, 0) if transpose_inputs1 else q_x_torch

            # There is no need to add the bias to the de-quantized values
            # as the bias is already included in the output quantizer
            # zero-point, in the analytical calibration
            q_w = torch.from_numpy(qweight).to(q_x_torch.device)
            mm = torch.matmul(q_x_torch, q_w)
            return out_quantizer.dequant(mm)

        # Ensure we have a single QuantizedGemm layer.
        assert len(q_module.quant_layers_dict) == 1, "Expected exactly one layer in QuantizedModule"
        assert isinstance(
            next(iter(q_module.quant_layers_dict.values()))[1], QuantizedGemm
        ), "Expected QuantizedGemm layer"

        _, quantized_layer = next(iter(q_module.quant_layers_dict.items()))
        device = x.device

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4711
        # Static per-channel weight quantization.
        weight_q, weight_scale, weight_zp, sum_w = self._per_channel_weight_quantization(
            weight, q_module, device
        )

        # Dynamic quantization for inputs.
        x_q, x_scale, x_zp, original_shape = self._dynamic_input_quantization(
            x, q_module, transpose_inputs1
        )

        # Perform integer matrix multiplication.
        mm_raw = torch.matmul(x_q, weight_q).long()
        k = weight_q.shape[0]

        # Apply correction and de-quantization.
        out_float = self._apply_correction_and_dequantize(
            mm_raw, x_q, x_zp, weight_zp, sum_w, k, x_scale, weight_scale
        )
        if len(original_shape) > 2:
            out_float = out_float.view(*original_shape[:-1], -1)

        # Add bias if available
        out_float = self._add_bias(out_float, quantized_layer, device)

        return out_float

    def _forward_fhe_static(
        self,
        x: torch.Tensor,
        q_module: QuantizedModule,
        transpose_inputs1: bool,
        qweight: numpy.ndarray,
    ) -> torch.Tensor:
        """FHE execution using static quantization.

        Args:
            x: Input tensor
            q_module: Quantized module containing quantization parameters
            transpose_inputs1: Whether to transpose the input
            qweight: Quantized weight matrix

        Returns:
            torch.Tensor: Output tensor after applying the linear operation in FHE mode
        """

        if self.private_key is None:
            self.keygen()  # pragma: no cover

        x_device = x.device
        q_x = q_module.quantize_input(x.cpu().numpy())
        assert q_x is not None
        assert isinstance(q_x, numpy.ndarray)
        q_x = numpy.transpose(q_x) if transpose_inputs1 else q_x

        # Need to slice the last GLWE (this will be improved in later  cml-extensions)
        num_valid_glwe_values_in_last_ciphertext = (
            qweight.shape[1] % self.poly_size or self.poly_size
        )

        # The GLWE backend needs uint64 encoding for both neg/pos values
        # Convert weights to required type.
        qweight = qweight.astype(numpy.int64).astype(numpy.uint64)

        # Some models have (B, C, H)-size activations,
        # for example LLMs: B=batch size, C=context length, H=hidden dime
        # while other models only have (B, H)-size activations.
        # Add a B=1 dimension if needed
        return_2d = False
        if q_x.ndim == 2:
            return_2d = True
            q_x = numpy.expand_dims(q_x, 0)

        # The GLWE backend needs contiguous memory uint64 encoding for both neg/pos values
        q_x = numpy.ascontiguousarray(q_x.astype(numpy.uint64))
        assert qweight.ndim == 2

        result_buffer = numpy.zeros(
            (q_x.shape[0], q_x.shape[1], qweight.shape[1]), dtype=numpy.int64
        )

        for idx, q_x_sample in enumerate(q_x):
            ciphertext = self.fhext.encrypt_matrix(  # pylint: disable=no-member
                pkey=self.private_key,
                crypto_params=self.glwe_crypto_params,
                data=q_x_sample,
            )
            encrypted_result = self.fhext.matrix_multiplication(
                encrypted_matrix=ciphertext,
                data=qweight,
                compression_key=self.compression_key,
            )
            q_result = self.fhext.decrypt_matrix(  # pylint: disable=no-member
                encrypted_result,
                self.private_key,
                self.glwe_crypto_params,
                num_valid_glwe_values_in_last_ciphertext,
            )
            result_buffer[idx, :] = q_result.astype(numpy.int64)

        # There is no need to add the bias to the de-quantized values
        # as the bias is already included in the output quantizer
        # zero-point, in the analytical calibration
        y = q_module.dequantize_output(*to_tuple(result_buffer))
        assert isinstance(y, numpy.ndarray)
        if return_2d:
            y = numpy.squeeze(y, axis=0)
        return torch.Tensor(y.astype(numpy.float32)).to(x_device)

    def _forward_fhe_dynamic(  # pylint: disable=too-many-locals
        self,
        x: torch.Tensor,
        q_module: QuantizedModule,
        transpose_inputs1: bool,
        weight: numpy.ndarray,
    ) -> torch.Tensor:
        """FHE execution using dynamic quantization.

        Args:
            x: Input tensor
            q_module: Quantized module containing quantization parameters
            transpose_inputs1: Whether to transpose the first input
            weight: Weight matrix in full precision

        Returns:
            torch.Tensor: Output tensor after applying the linear operation in FHE mode
        """
        if self.private_key is None:
            self.keygen()  # pragma: no cover

        device = x.device

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4711
        # Dynamic quantization for weights and input.
        weight_q, weight_scale, weight_zp, sum_w = self._per_channel_weight_quantization(
            weight, q_module, device
        )
        x_q, x_scale, x_zp, original_shape = self._dynamic_input_quantization(
            x, q_module, transpose_inputs1
        )

        # The GLWE backend needs uint64 encoding for both neg/pos values
        # Convert quantized data to numpy arrays for encryption.
        x_q_int = x_q.long().cpu().numpy().astype(numpy.int64).astype(numpy.uint64)
        weight_q_int = weight_q.long().cpu().numpy().astype(numpy.int64).astype(numpy.uint64)

        # Ensure x_q_int is 3D.
        if x_q_int.ndim == 2:
            x_q_int = numpy.expand_dims(x_q_int, 0)

        num_valid_glwe_values_in_last_ciphertext = (
            weight_q_int.shape[1] % self.poly_size or self.poly_size
        )
        batch, n_rows, _ = x_q_int.shape
        result_buffer = numpy.zeros((batch, n_rows, weight_q_int.shape[1]), dtype=numpy.int64)

        for idx, q_x_sample in enumerate(x_q_int):
            ciphertext = self.fhext.encrypt_matrix(  # pylint: disable=no-member
                pkey=self.private_key,
                crypto_params=self.glwe_crypto_params,
                data=q_x_sample,
            )
            encrypted_result = self.fhext.matrix_multiplication(
                encrypted_matrix=ciphertext,
                data=weight_q_int,
                compression_key=self.compression_key,
            )
            q_result = self.fhext.decrypt_matrix(  # pylint: disable=no-member
                encrypted_result,
                self.private_key,
                self.glwe_crypto_params,
                num_valid_glwe_values_in_last_ciphertext,
            )
            result_buffer[idx, :] = q_result.astype(numpy.int64)

        result_tensor = torch.tensor(result_buffer, device=device, dtype=torch.long)
        k = weight_q.shape[0]
        out_tensor = self._apply_correction_and_dequantize(
            result_tensor, x_q, x_zp, weight_zp, sum_w, k, x_scale, weight_scale
        )

        out_tensor = (
            out_tensor.view(*original_shape[:-1], -1) if original_shape[:-1] else out_tensor
        )

        assert (
            original_shape[:-1] == out_tensor.shape[:-1]
        ), "Original shape and output shape do not match"
        _, quantized_layer = next(iter(q_module.quant_layers_dict.items()))
        out_tensor = self._add_bias(out_tensor, quantized_layer, device)
        return out_tensor

    def forward(
        self,
        x: torch.Tensor,
        q_module: QuantizedModule,
        fhe: HybridFHEMode,
    ) -> torch.Tensor:
        """Perform the inference of this linear layer.

        Args:
            x (torch.Tensor): Input or previous layer activations.
            q_module (QuantizedModule): Module containing the quantized layer.
            fhe (HybridFHEMode): Execution mode, can be 'DISABLE' or 'EXECUTE'.

        Returns:
            torch.Tensor: The result of applying the linear layer.
        """
        layers_in_module = list(q_module.quant_layers_dict.values())
        assert len(layers_in_module) == 1, "Expected exactly one linear layer in QuantizedModule"

        quantized_linear_op = layers_in_module[0][1]
        assert quantized_linear_op.supported_by_linear_backend()

        transpose_inputs1 = quantized_linear_op.attrs.get("transA", False)
        transpose_inputs2 = quantized_linear_op.attrs.get("transB", False)

        # Extract weight parameters.
        weight, qweight = self._extract_weight_params(quantized_linear_op, transpose_inputs2)

        if fhe == HybridFHEMode.DISABLE:
            return self._forward_clear(x, q_module, transpose_inputs1, qweight, weight)

        if self.use_dynamic_quantization:
            return self._forward_fhe_dynamic(x, q_module, transpose_inputs1, weight)

        return self._forward_fhe_static(x, q_module, transpose_inputs1, qweight)
