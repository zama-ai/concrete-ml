"""QuantizedModule API."""
import copy
from typing import Optional, Union

import numpy
from concrete.common.compilation.artifacts import CompilationArtifacts
from concrete.common.compilation.configuration import CompilationConfiguration
from concrete.common.fhe_circuit import FHECircuit
from concrete.numpy.np_fhe_compiler import NPFHECompiler

from ..common.debugging import assert_true
from .quantized_array import QuantizedArray
from .quantized_ops import QuantizedOp


class QuantizedModule:
    """Inference for a quantized model."""

    quant_layers_dict: dict
    _mode: str
    q_input: Optional[QuantizedArray]
    forward_fhe: Union[None, FHECircuit]

    def __init__(self, quant_layers_dict: dict):
        self.quant_layers_dict = copy.deepcopy(quant_layers_dict)
        self.compiled = False
        self.forward_fhe = None
        self.q_input = None

    def __call__(self, x: numpy.ndarray):
        return self.forward(x)

    def forward(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only.

        Args:
            qvalues (numpy.ndarray): numpy.array containing the quantized values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        # Make sure that the input is quantized
        assert_true(
            qvalues.dtype == numpy.uint8,
            on_error_msg=f"qvalues.dtype={qvalues.dtype} is not uint8. "
            f"Make sure you quantize your input before calling forward.",
            error_type=ValueError,
        )

        return self._forward(qvalues)

    def _forward(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Forward function for the FHE circuit.

        Args:
            qvalues (numpy.ndarray): numpy.array containing the quantized values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        # satisfy mypy
        assert self.q_input is not None, "q_input is not set"
        self.q_input.update_quantized_values(qvalues)
        q_array = self.q_input

        for _, layer in self.quant_layers_dict.items():
            q_array = layer(q_array)

        # mypy compliance
        assert isinstance(q_array, QuantizedArray)

        return q_array.qvalues

    def forward_and_dequant(self, q_x: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only plus dequantization.

        Args:
            q_x (Union[numpy.ndarray, QuantizedArray]): QuantizedArray containing the inputs
                                                        or a numpy.array containing the q_values.
                                                        In the latter, the stored input parameters
                                                        are used:
                                                        (q_input.scale, q_input.zero_point).

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        q_out = self.forward(q_x)
        return self.dequantize_output(q_out)

    def quantize_input(self, values: numpy.ndarray) -> numpy.ndarray:
        """Take the inputs in fp32 and quantize it using the learned quantization parameters.

        Args:
            values (numpy.ndarray): Floating point values.

        Returns:
            numpy.ndarray: Quantized (numpy.uint8) values.
        """
        # satisfy mypy
        assert self.q_input is not None
        qvalues = self.q_input.update_values(values)
        return qvalues.astype(numpy.uint8)

    def dequantize_output(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Take the last layer q_out and use its dequant function.

        Args:
            qvalues (numpy.ndarray): Quantized values of the last layer.

        Returns:
            numpy.ndarray: Dequantized values of the last layer.
        """
        last_layer = list(self.quant_layers_dict.values())[-1]
        real_values = (
            QuantizedArray(
                last_layer.n_bits,
                qvalues,
                value_is_float=False,
                scale=last_layer.output_scale,
                zero_point=last_layer.output_zero_point,
            ).dequant()
            if isinstance(last_layer, QuantizedOp)
            else last_layer.q_out.update_quantized_values(qvalues)
        )
        return real_values

    def compile(
        self,
        q_input: QuantizedArray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
    ) -> FHECircuit:
        """Compile the forward function of the module.

        Args:
            q_input (QuantizedArray): Needed for tracing and building the boundaries.
            compilation_configuration (Optional[CompilationConfiguration]): Configuration object
                                                                            to use during
                                                                            compilation
            compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill during
                                                                    compilation
            show_mlir (bool): if set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.

        Returns:
            FHECircuit: the compiled FHECircuit.
        """

        self.q_input = copy.deepcopy(q_input)
        compiler = NPFHECompiler(
            self._forward,
            {
                "qvalues": "encrypted",
            },
            compilation_configuration,
            compilation_artifacts,
        )
        self.forward_fhe = compiler.compile_on_inputset(
            (numpy.expand_dims(arr, 0) for arr in self.q_input.qvalues), show_mlir
        )

        return self.forward_fhe
