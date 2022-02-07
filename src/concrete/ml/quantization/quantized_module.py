"""QuantizedModule API."""
import copy
from typing import Dict, Iterable, Optional, Tuple, Union

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

    ordered_module_input_names: Tuple[str, ...]
    ordered_module_output_names: Tuple[str, ...]
    quant_layers_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    q_input: Optional[QuantizedArray]
    forward_fhe: Union[None, FHECircuit]

    def __init__(
        self,
        ordered_module_input_names: Iterable[str],
        ordered_module_output_names: Iterable[str],
        quant_layers_dict: dict,
    ):
        self.ordered_module_input_names = tuple(ordered_module_input_names)
        self.ordered_module_output_names = tuple(ordered_module_output_names)

        assert_true(
            (num_inputs := len(self.ordered_module_input_names)) == 1,
            f"{QuantizedModule.__class__.__name__} only supports a single input for now, "
            f"got {num_inputs}",
        )

        assert_true(
            (num_outputs := len(self.ordered_module_output_names)) == 1,
            f"{QuantizedModule.__class__.__name__} only supports a single output for now, "
            f"got {num_outputs}",
        )

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
            issubclass(qvalues.dtype.type, numpy.integer),
            on_error_msg=f"qvalues.dtype={qvalues.dtype} is not an integer type. "
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

        # Init layer_results with the inputs
        layer_results = dict(zip(self.ordered_module_input_names, (self.q_input,)))

        for output_name, (input_names, layer) in self.quant_layers_dict.items():
            inputs = (layer_results[input_name] for input_name in input_names)
            output = layer(*inputs)
            layer_results[output_name] = output

        outputs = tuple(
            layer_results[output_name] for output_name in self.ordered_module_output_names
        )

        assert_true(len(outputs) == 1)

        return outputs[0].qvalues

    def forward_and_dequant(self, q_x: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only plus dequantization.

        Args:
            q_x (numpy.ndarray): numpy.ndarray containing the quantized input values. Requires the
                input dtype to be uint8.

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
        assert self.q_input is not None, "q_input is not set"
        qvalues = self.q_input.update_values(values)
        assert_true(
            numpy.array_equal(qvalues.astype(numpy.uint32), qvalues),
            on_error_msg="Input quantizer does not give values within uint32.",
        )
        return qvalues.astype(numpy.uint32)

    def dequantize_output(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Take the last layer q_out and use its dequant function.

        Args:
            qvalues (numpy.ndarray): Quantized values of the last layer.

        Returns:
            numpy.ndarray: Dequantized values of the last layer.
        """
        output_layers = (
            self.quant_layers_dict[output_name][1]
            for output_name in self.ordered_module_output_names
        )
        real_values = tuple(
            QuantizedArray(
                output_layer.n_bits,
                qvalues,
                value_is_float=False,
                scale=output_layer.output_scale,
                zero_point=output_layer.output_zero_point,
            ).dequant()
            for output_layer in output_layers
        )

        assert_true(len(real_values) == 1)

        return real_values[0]

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
