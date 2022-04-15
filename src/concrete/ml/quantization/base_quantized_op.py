"""Base Quantized Op class that implements quantization for a float numpy op."""
from abc import ABC
from copy import deepcopy
from inspect import Parameter, _empty, signature
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import numpy

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
