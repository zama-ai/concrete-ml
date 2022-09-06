"""QuantizedModule API."""
import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy
from concrete.numpy.compilation.artifacts import DebugArtifacts
from concrete.numpy.compilation.circuit import Circuit
from concrete.numpy.compilation.compiler import Compiler
from concrete.numpy.compilation.configuration import Configuration

from ..common.debugging import assert_true
from ..common.utils import DEFAULT_P_ERROR_PBS, generate_proxy_function
from .base_quantized_op import QuantizedOp
from .quantizers import QuantizedArray, UniformQuantizer


class QuantizedModule:
    """Inference for a quantized model."""

    ordered_module_input_names: Tuple[str, ...]
    ordered_module_output_names: Tuple[str, ...]
    quant_layers_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    input_quantizers: List[UniformQuantizer]
    output_quantizers: List[UniformQuantizer]
    forward_fhe: Union[None, Circuit]

    def __init__(
        self,
        ordered_module_input_names: Iterable[str] = None,
        ordered_module_output_names: Iterable[str] = None,
        quant_layers_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]] = None,
    ):
        # If any of the arguments are not provided, skip the init
        if not all([ordered_module_input_names, ordered_module_output_names, quant_layers_dict]):
            return

        # for mypy
        assert isinstance(ordered_module_input_names, Iterable)
        assert isinstance(ordered_module_output_names, Iterable)
        assert all([ordered_module_input_names, ordered_module_output_names, quant_layers_dict])
        self.ordered_module_input_names = tuple(ordered_module_input_names)
        self.ordered_module_output_names = tuple(ordered_module_output_names)

        assert_true(
            (num_outputs := len(self.ordered_module_output_names)) == 1,
            f"{QuantizedModule.__class__.__name__} only supports a single output for now, "
            f"got {num_outputs}",
        )

        assert quant_layers_dict is not None
        self.quant_layers_dict = copy.deepcopy(quant_layers_dict)
        self._is_compiled = False
        self.forward_fhe = None
        self.input_quantizers = []
        self.output_quantizers = self._set_output_quantizers()
        self._onnx_model = None
        self._post_processing_params: Dict[str, Any] = {}

    @property
    def is_compiled(self) -> bool:
        """Return the compiled status of the module.

        Returns:
            bool: the compiled status of the module.
        """
        return self._is_compiled

    @property
    def fhe_circuit(self) -> Circuit:
        """Get the FHE circuit.

        Returns:
            Circuit: the FHE circuit
        """
        return self.forward_fhe

    @fhe_circuit.setter
    def fhe_circuit(self, fhe_circuit: Circuit):
        """Set the FHE circuit.

        Args:
            fhe_circuit (Circuit): the FHE circuit
        """
        self.forward_fhe = fhe_circuit
        self._is_compiled = True

    @property
    def post_processing_params(self) -> Dict[str, Any]:
        """Get the post-processing parameters.

        Returns:
            Dict[str, Any]: the post-processing parameters
        """
        return self._post_processing_params

    @post_processing_params.setter
    def post_processing_params(self, post_processing_params: Dict[str, Any]):
        """Set the post-processing parameters.

        Args:
            post_processing_params (dict): the post-processing parameters
        """
        self._post_processing_params = post_processing_params

    def post_processing(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Post-processing of the quantized output.

        Args:
            qvalues (numpy.ndarray): numpy.ndarray containing the quantized input values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        return self.dequantize_output(qvalues)

    def _set_output_quantizers(self) -> List[UniformQuantizer]:
        """Get the output quantizers.

        Returns:
            List[UniformQuantizer]: List of output quantizers.
        """
        output_layers = (
            self.quant_layers_dict[output_name][1]
            for output_name in self.ordered_module_output_names
        )
        output_quantizers = list(
            QuantizedArray(
                output_layer.n_bits,
                values=None,
                value_is_float=False,
                stats=output_layer.output_quant_stats,
                params=output_layer.output_quant_params,
            ).quantizer
            for output_layer in output_layers
        )
        return output_quantizers

    @property
    def onnx_model(self):
        """Get the ONNX model.

        .. # noqa: DAR201

        Returns:
           _onnx_model (onnx.ModelProto): the ONNX model
        """
        return self._onnx_model

    @onnx_model.setter
    def onnx_model(self, value):
        self._onnx_model = value

    def __call__(self, *x: numpy.ndarray):
        return self.forward(*x)

    def forward(self, *qvalues: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only.

        Args:
            *qvalues (numpy.ndarray): numpy.array containing the quantized values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        # Make sure that the input is quantized
        assert_true(
            len(
                invalid_inputs := tuple(
                    (idx, qvalue)
                    for idx, qvalue in enumerate(qvalues)
                    if not issubclass(qvalue.dtype.type, numpy.integer)
                )
            )
            == 0,
            f"Inputs: {', '.join(f'#{val[0]} ({val[1].dtype})' for val in invalid_inputs)} are not "
            "integer types. Make sure you quantize your input before calling forward.",
            ValueError,
        )

        return self._forward(*qvalues)

    def _forward(self, *qvalues: numpy.ndarray) -> numpy.ndarray:
        """Forward function for the FHE circuit.

        Args:
            *qvalues (numpy.ndarray): numpy.array containing the quantized values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """

        assert_true(
            (n_qvalues := len(qvalues)) == (n_qinputs := len(self.input_quantizers)),
            f"Got {n_qvalues} inputs, expected {n_qinputs}",
            TypeError,
        )

        q_inputs = [
            QuantizedArray(
                self.input_quantizers[idx].n_bits,
                qvalues[idx],
                value_is_float=False,
                options=self.input_quantizers[idx].quant_options,
                stats=self.input_quantizers[idx].quant_stats,
                params=self.input_quantizers[idx].quant_params,
            )
            for idx in range(len(self.input_quantizers))
        ]

        # Init layer_results with the inputs
        layer_results = dict(zip(self.ordered_module_input_names, q_inputs))

        for output_name, (input_names, layer) in self.quant_layers_dict.items():
            inputs = (layer_results[input_name] for input_name in input_names)
            output = layer(*inputs)
            layer_results[output_name] = output

        outputs = tuple(
            layer_results[output_name] for output_name in self.ordered_module_output_names
        )

        assert_true(len(outputs) == 1)

        return outputs[0].qvalues

    def forward_and_dequant(self, *q_x: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only plus dequantization.

        Args:
            *q_x (numpy.ndarray): numpy.ndarray containing the quantized input values. Requires the
                input dtype to be uint8.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        q_out = self.forward(*q_x)
        return self.dequantize_output(q_out)

    def quantize_input(
        self, *values: numpy.ndarray
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Take the inputs in fp32 and quantize it using the learned quantization parameters.

        Args:
            *values (numpy.ndarray): Floating point values.

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: Quantized (numpy.uint32) values.
        """

        assert_true(
            (n_values := len(values)) == (n_qinputs := len(self.input_quantizers)),
            f"Got {n_values} inputs, expected {n_qinputs}",
            TypeError,
        )

        qvalues = tuple(self.input_quantizers[idx].quant(values[idx]) for idx in range(len(values)))

        qvalues_as_uint32 = tuple(qvalue.astype(numpy.uint32) for qvalue in qvalues)
        assert_true(
            all(
                numpy.array_equal(qval_uint32, qval)
                for qval_uint32, qval in zip(qvalues_as_uint32, qvalues)
            ),
            on_error_msg="Input quantizer does not give values within uint32.",
        )
        return qvalues_as_uint32[0] if len(qvalues_as_uint32) == 1 else qvalues_as_uint32

    def dequantize_output(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Take the last layer q_out and use its dequant function.

        Args:
            qvalues (numpy.ndarray): Quantized values of the last layer.

        Returns:
            numpy.ndarray: Dequantized values of the last layer.
        """
        real_values = tuple(
            output_quantizer.dequant(qvalues) for output_quantizer in self.output_quantizers
        )

        assert_true(len(real_values) == 1)

        return real_values[0]

    def set_inputs_quantization_parameters(self, *input_q_params: UniformQuantizer):
        """Set the quantization parameters for the module's inputs.

        Args:
            *input_q_params (UniformQuantizer): The quantizer(s) for the module.
        """
        assert_true(
            (n_values := len(input_q_params)) == (n_inputs := len(self.ordered_module_input_names)),
            f"Got {n_values} inputs, expected {n_inputs}",
            TypeError,
        )

        self.input_quantizers.clear()
        self.input_quantizers.extend(copy.deepcopy(q_params) for q_params in input_q_params)

    def compile(
        self,
        q_inputs: Union[Tuple[numpy.ndarray, ...], numpy.ndarray],
        configuration: Optional[Configuration] = None,
        compilation_artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
        p_error: Optional[float] = DEFAULT_P_ERROR_PBS,
    ) -> Circuit:
        """Compile the forward function of the module.

        Args:
            q_inputs (Union[Tuple[numpy.ndarray, ...], numpy.ndarray]): Needed for tracing and
                building the boundaries.
            configuration (Optional[Configuration]): Configuration object to use during compilation
            compilation_artifacts (Optional[DebugArtifacts]): Artifacts object to fill during
            show_mlir (bool): if set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.
            use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
                Defaults to False.
            p_error (Optional[float]): probability of error of a PBS.

        Returns:
            Circuit: the compiled Circuit.
        """

        if not isinstance(q_inputs, tuple):
            q_inputs = (q_inputs,)
        else:
            ref_len = q_inputs[0].shape[0]
            assert_true(
                all(q_input.shape[0] == ref_len for q_input in q_inputs),
                "Mismatched dataset lengths",
            )

        # concrete-numpy does not support variable *args-style functions, so compile a proxy
        # function dynamically with a suitable number of arguments
        forward_proxy, orig_args_to_proxy_func_args = generate_proxy_function(
            self._forward, self.ordered_module_input_names
        )

        compiler = Compiler(
            forward_proxy,
            {arg_name: "encrypted" for arg_name in orig_args_to_proxy_func_args.values()},
        )

        def get_inputset_iterable():
            if len(self.input_quantizers) > 1:
                return (
                    tuple(numpy.expand_dims(q_input[idx], 0) for q_input in q_inputs)
                    for idx in range(ref_len)
                )
            return (numpy.expand_dims(arr, 0) for arr in q_inputs[0])

        inputset = get_inputset_iterable()

        self.forward_fhe = compiler.compile(
            inputset,
            configuration,
            compilation_artifacts,
            show_mlir=show_mlir,
            virtual=use_virtual_lib,
            p_error=p_error,
        )

        self._is_compiled = True

        return self.forward_fhe
