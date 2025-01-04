"""GLWE backend for some supported layers."""

import json

import numpy
import torch

from ..common.utils import HybridFHEMode, to_tuple
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
    except ImportError:
        return False


class GLWELinearLayerExecutor:
    """GLWE execution helper for pure linear layers."""

    def __init__(
        self,
        private_key=None,
        compression_key=None,
    ):
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

    def keygen(self):
        """Generate private and compression key."""
        # pylint: disable-next=no-member
        self.private_key, self.compression_key = self.fhext.create_private_key(
            self.glwe_crypto_params
        )

    def _forward_clear(
        self,
        x: torch.Tensor,
        q_module: QuantizedModule,
        transpose_inputs1: bool,
        q_weight: numpy.ndarray,
    ) -> torch.Tensor:
        quantizer = TorchUniformQuantizer(q_module.input_quantizers[0])
        out_quantizer = TorchUniformQuantizer(q_module.output_quantizers[0])

        q_x_torch = quantizer.quant(x, dtype=torch.float32)
        q_x_torch = torch.transpose(q_x_torch, 1, 0) if transpose_inputs1 else q_x_torch

        # There is no need to add the bias to the de-quantized values
        # as the bias is already included in the output quantizer
        # zero-point, in the analytical calibration
        q_w = torch.from_numpy(q_weight).to(q_x_torch.device)
        mm = torch.matmul(q_x_torch, q_w)
        return out_quantizer.dequant(mm)

    def forward(
        self, x: torch.Tensor, q_module: QuantizedModule, fhe: HybridFHEMode
    ) -> torch.Tensor:
        """Perform the inference of this linear layer.

        Args:
            x (numpy.ndarray): inputs or previous layer activations
            q_module (QuantizedModule): quantized module that contains the layer which
                is executed through this helper. Stores quantized weights and input
                and output quantizers
            fhe (HybridFHEMode): execution mode, can be 'disable' or 'execute'

        Returns:
            numpy.ndarray: result of applying the linear layer
        """

        # Extract all the layers in this quantized module
        # and check that there is only one, as only a single linear layer QM
        # can be optimized
        layers_in_module = list(q_module.quant_layers_dict.values())
        assert len(layers_in_module) == 1

        # Get the single linear op in this module
        quantized_linear_op = layers_in_module[0][1]
        assert quantized_linear_op.supported_by_linear_backend()

        # Use default false so we also support MatMul impl, MatMul does not have these flags
        transpose_inputs1 = quantized_linear_op.attrs.get("transA", False)
        transpose_inputs2 = quantized_linear_op.attrs.get("transB", False)

        # Extract the weights and bias in this single linear layer
        weight_bias = list(quantized_linear_op.constant_inputs.values())

        # Make sure the weights used symmetric quantization
        assert weight_bias[0].quantizer.quant_params.zero_point == 0

        # Retrieve quantized weights
        q_weight = weight_bias[0].values
        assert isinstance(q_weight, numpy.ndarray)
        assert q_weight.dtype == numpy.float32

        q_weight = numpy.transpose(q_weight) if transpose_inputs2 else q_weight

        if fhe == HybridFHEMode.DISABLE:
            return self._forward_clear(x, q_module, transpose_inputs1, q_weight)

        if self.private_key is None:
            self.keygen()  # pragma: no cover

        x_device = x.device
        q_x = q_module.quantize_input(x.cpu().numpy())
        assert q_x is not None
        assert isinstance(q_x, numpy.ndarray)

        q_x = numpy.transpose(q_x) if transpose_inputs1 else q_x

        # Need to slice the last GLWE (this will be improved in later cml-extensions)
        num_valid_glwe_values_in_last_ciphertext = (
            q_weight.shape[1] % self.poly_size or self.poly_size
        )

        # The GLWE backend needs uint64 encoding for both neg/pos values
        q_weight = q_weight.astype(numpy.uint64)

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

        assert q_weight.ndim == 2
        result_buffer = numpy.zeros(
            (q_x.shape[0], q_x.shape[1], q_weight.shape[1]), dtype=numpy.int64
        )

        for idx, q_x_sample in enumerate(q_x):

            ciphertext = self.fhext.encrypt_matrix(  # pylint: disable=no-member
                pkey=self.private_key, crypto_params=self.glwe_crypto_params, data=q_x_sample
            )
            encrypted_result = self.fhext.matrix_multiplication(  # pylint: disable=no-member
                encrypted_matrix=ciphertext,
                data=q_weight.astype(numpy.uint64),
                compression_key=self.compression_key,
            )
            q_result = self.fhext.decrypt_matrix(  # pylint: disable=no-member
                encrypted_result,
                self.private_key,
                self.glwe_crypto_params,
                num_valid_glwe_values_in_last_ciphertext,
            )
            q_result = q_result.astype(numpy.int64)

            result_buffer[idx, :] = q_result

        # There is no need to add the bias to the de-quantized values
        # as the bias is already included in the output quantizer
        # zero-point, in the analytical calibration
        y = q_module.dequantize_output(*to_tuple(result_buffer))

        assert isinstance(y, numpy.ndarray)

        if return_2d:
            y = numpy.squeeze(y)

        return torch.Tensor(y.astype(numpy.float32)).to(x_device)
