"""Base Quantized Op class that implements quantization for a float numpy op."""

from copy import deepcopy
from inspect import Parameter, _empty, signature
from typing import Any, Callable, Dict, List, Optional, Set, TextIO, Tuple, Type, Union, cast

import numpy

from concrete import fhe

from ..common.debugging import assert_false, assert_true
from ..common.serialization.dumpers import dump, dumps
from ..common.utils import compute_bits_precision
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL
from ..onnx.ops_impl import ONNXMixedFunction, RawOpOutput
from .quantizers import (
    MinMaxQuantizationStats,
    QuantizationOptions,
    QuantizedArray,
    UniformQuantizationParameters,
)

ONNXOpInputOutputType = Union[numpy.ndarray, QuantizedArray, None]

ALL_QUANTIZED_OPS: Set[Type] = set()

ONNX_OPS_TO_QUANTIZED_IMPL: Dict[str, Type["QuantizedOp"]] = {}

# This constant determines the number of bits for the quantization of input and output values
# of an ML model. This is not necessarily the maximum bit-width in the network, as Gemm/Conv ops
# have output bit-width that is related to their weights and inputs.
# Run time in FHE is strongly impacted by the number of bits, with increases of 5-20x for
# each additional bit. However, strong quantization of inputs and outputs can negatively impact
# accuracy. This value is chosen as a compromise between run time and model accuracy. This default
# value is used only if the user does not specifically specify a value for input or output
# bit-width.
DEFAULT_MODEL_BITS = 5


class QuantizedOp:
    """Base class for quantized ONNX ops implemented in numpy.

    Args:
        n_bits_output (int): The number of bits to use for the quantization of the output
        op_instance_name (str): The name that should be assigned to this operation, used
            to retrieve it later or get debugging information about this op (bit-width, value range,
            integer intermediary values, op-specific error messages). Usually this name is the same
            as the ONNX operation name for which this operation is constructed.
        int_input_names (Set[str]): The set of names of integer tensors that are inputs to this op
        constant_inputs (Optional[Union[Dict[str, Any], Dict[int, Any]]]): The constant tensors
            that are inputs to this op
        input_quant_opts (QuantizationOptions): Input quantizer options, determine the quantization
            that is applied to input tensors (that are not constants)
    """

    # impl is not optional but mypy has a long standing bug and is not able to understand this
    # properly. See https://github.com/python/mypy/issues/708#issuecomment-605636623
    impl: Optional[Callable[..., Tuple[numpy.ndarray, ...]]] = None
    n_bits: int

    # Quantized Ops have a quantizer for the input and one for the output
    # For the output we store quantizer statistics and computed parameters
    # For the input, we store only the quantization options, as the statistics
    # are taken from the previous op's output and the input quantization parameters are computed
    # from the input quantization options and the previous op's output statistics
    output_quant_params: Optional[UniformQuantizationParameters]
    output_quant_stats: Optional[MinMaxQuantizationStats]
    input_quant_opts: QuantizationOptions

    constant_inputs: Dict[int, Any]
    attrs: Dict[str, Any] = {}
    _authorized_attr_names: Set[str] = set()
    # This can be used for custom implementations of some missing or buggy operators
    _impl_for_op_named: Optional[str] = None
    _default_attrs: Dict[str, Any] = {}
    _params_name_to_input_idx: Dict[str, int] = {}
    _input_idx_to_params_name: Dict[int, str] = {}
    _params_that_are_onnx_inputs: Set[str] = set()
    _params_that_are_onnx_var_inputs: Set[str] = set()
    _params_that_are_required_onnx_inputs: Set[str] = set()
    _has_attr: bool
    _inputs_not_quantized: Set[str] = set()
    quantize_inputs_with_model_outputs_precision: bool = False

    # The ONNX name of this op instance (e.g., "Conv_9", "MatMul_5" etc.)
    op_instance_name: str

    # Determines if this op computes a tensor that is a graph output, i.e., a tensor
    # that will be decrypted and de-quantized in the clear
    produces_graph_output = False

    # Determines if the op produces a raw output (not quantized). This can
    # be a float or integer tensor that contains non-encrypted values
    produces_raw_output = False

    error_tracker: Optional[List[int]] = None
    debug_value_tracker: Optional[
        Dict[str, Dict[Union[int, str], Optional[ONNXOpInputOutputType]]]
    ] = None

    POSITIONAL_ARGUMENTS_KINDS = {
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    }
    VAR_POSITIONAL_ARGUMENTS_KINDS = {Parameter.VAR_POSITIONAL}

    def __init__(
        self,
        n_bits_output: int,
        op_instance_name: str,
        int_input_names: Optional[Set[str]] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        self.n_bits = n_bits_output

        if input_quant_opts is not None:
            self.input_quant_opts = deepcopy(input_quant_opts)
        else:
            # By default, if the input quantization options are not given
            # make the op work in legacy mode, where inputs and outputs are quantized
            # to the same number of bits
            self.input_quant_opts = QuantizationOptions(self.n_bits, is_signed=True)

        self.output_quant_params = None
        self.output_quant_stats = None

        # By default, such as for operators that only have a float implementation,
        # we assume a single integer input tensor. Since we can't use {"0"} as a default value in
        # Python, we use None and we initialize the set to {"0"}. When constructing the ops
        # through ONNX -> NumpyModule conversion, a value should always be provided
        # for int_input_names. This default is only for instantiating ops manually, which is not
        # recommended usage. We use "0" since this is a common ONNX tensor name for inputs.
        self._int_input_names = {"0"} if int_input_names is None else int_input_names

        constant_inputs_per_name: Dict[str, Any] = {}

        def _convert_input_name_or_idx_to_input_name(input_name_or_idx: Union[str, int]) -> str:
            if isinstance(input_name_or_idx, str):

                # When serializing constant inputs using a JSONEncoder, integer indexes are
                # automatically converted to strings. We therefore need to put these indexes back
                # to integers by checking if the current string can be converted to an integer
                # object.
                if input_name_or_idx.isdigit():
                    input_name_or_idx = int(input_name_or_idx)
                else:
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

        invalid_input_names = constant_inputs_per_name.keys() - self._params_that_are_onnx_inputs
        assert_true(
            len(invalid_input_names) == 0,
            "Got invalid constant input names or indices: "
            f"{', '.join(sorted(invalid_input_names))}.\n"
            f"Valid input names: {(', '.join(sorted(self._params_that_are_onnx_inputs)))}.",
        )

        # Convert input names to input indices
        self.constant_inputs = {
            self._params_name_to_input_idx[input_name]: constant_value
            for input_name, constant_value in constant_inputs_per_name.items()
        }
        unknown_attrs = attrs.keys() - self._authorized_attr_names
        assert_true(
            len(unknown_attrs) == 0,
            f"Got the following unknown attributes: {', '.join(sorted(unknown_attrs))}. "
            + (
                f"Accepted attributes: {', '.join(sorted(self._authorized_attr_names))}."
                if len(self._authorized_attr_names) > 0
                else f"{self.__class__.__name__} does not accept attributes."
            ),
        )

        self.attrs = dict(self._default_attrs, **deepcopy(attrs))

        # Only use QAT for layers that need it (that mix encrypted values: conv, dense, add, etc...)
        if self.can_fuse():
            self.input_quant_opts.is_qat = False

        # Set the operation's name, which is used to identify this op in the Concrete ML op graph
        # with respect to the ONNX graph (usually we keep use ONNX op name)
        self.op_instance_name = op_instance_name

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        metadata["_impl_for_op_named"] = self._impl_for_op_named

        # Attributes needed for instantiating a quantized operator
        metadata["n_bits"] = self.n_bits
        metadata["op_instance_name"] = self.op_instance_name
        metadata["_int_input_names"] = self.int_input_names
        metadata["constant_inputs"] = self.constant_inputs
        metadata["input_quant_opts"] = self.input_quant_opts
        metadata["attrs"] = self.attrs

        # Output quantization attributes
        metadata["output_quant_params"] = self.output_quant_params
        metadata["output_quant_stats"] = self.output_quant_stats
        if hasattr(self, "output_quant_opts"):
            metadata["output_quant_opts"] = self.output_quant_opts

        # QuantizedOp attributes
        # Set attributes are converted to list since set are not serializable. Additionally, in
        # order to be able to properly compare identical serialized objects, these lists are sorted
        metadata["_authorized_attr_names"] = self._authorized_attr_names
        metadata["_default_attrs"] = self._default_attrs
        metadata["_params_name_to_input_idx"] = self._params_name_to_input_idx
        metadata["_input_idx_to_params_name"] = self._input_idx_to_params_name
        metadata["_params_that_are_onnx_inputs"] = self._params_that_are_onnx_inputs
        metadata["_params_that_are_onnx_var_inputs"] = self._params_that_are_onnx_var_inputs
        metadata[
            "_params_that_are_required_onnx_inputs"
        ] = self._params_that_are_required_onnx_inputs
        metadata["_has_attr"] = self._has_attr
        metadata["_inputs_not_quantized"] = self._inputs_not_quantized
        metadata[
            "quantize_inputs_with_model_outputs_precision"
        ] = self.quantize_inputs_with_model_outputs_precision
        metadata["produces_graph_output"] = self.produces_graph_output
        metadata["produces_raw_output"] = self.produces_raw_output
        metadata["error_tracker"] = self.error_tracker
        metadata["debug_value_tracker"] = self.debug_value_tracker

        # Additional attributes specific to some quantized operators
        for attribute_name in vars(self):
            attribute_value = getattr(self, attribute_name)

            if attribute_name not in metadata:
                metadata[attribute_name] = attribute_value

        return metadata

    @staticmethod
    def load_dict(metadata: Dict):
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            QuantizedOp: The loaded object.
        """

        # Instantiate the quantized operator
        quantized_op = ONNX_OPS_TO_QUANTIZED_IMPL[metadata.pop("_impl_for_op_named")]
        obj = quantized_op(
            n_bits_output=metadata.pop("n_bits"),
            op_instance_name=metadata.pop("op_instance_name"),
            int_input_names=metadata.pop("_int_input_names"),
            constant_inputs=metadata.pop("constant_inputs"),
            input_quant_opts=metadata.pop("input_quant_opts"),
            **metadata.pop("attrs"),
        )

        # Output quantization attributes
        obj.output_quant_params = metadata.pop("output_quant_params")
        obj.output_quant_stats = metadata.pop("output_quant_stats")
        if "output_quant_opts" in metadata:
            assert hasattr(
                obj, "output_quant_opts"
            ), f"{obj.__class__.__name__} has no output_quant_opts attribute."

            obj.output_quant_opts = metadata.pop("output_quant_opts")  # type: ignore[arg-type]

        # QuantizedOp attributes
        # pylint: disable=protected-access
        obj._authorized_attr_names = metadata.pop("_authorized_attr_names")
        obj._default_attrs = metadata.pop("_default_attrs")
        obj._params_name_to_input_idx = metadata.pop("_params_name_to_input_idx")
        obj._input_idx_to_params_name = metadata.pop("_input_idx_to_params_name")
        obj._params_that_are_onnx_inputs = metadata.pop("_params_that_are_onnx_inputs")
        obj._params_that_are_onnx_var_inputs = metadata.pop("_params_that_are_onnx_var_inputs")
        obj._params_that_are_required_onnx_inputs = metadata.pop(
            "_params_that_are_required_onnx_inputs"
        )
        obj._has_attr = metadata.pop("_has_attr")
        obj._inputs_not_quantized = metadata.pop("_inputs_not_quantized")
        obj.quantize_inputs_with_model_outputs_precision = metadata.pop(
            "quantize_inputs_with_model_outputs_precision"
        )
        obj.produces_graph_output = metadata.pop("produces_graph_output")
        obj.produces_raw_output = metadata.pop("produces_raw_output")
        obj.error_tracker = metadata.pop("error_tracker")
        obj.debug_value_tracker = metadata.pop("debug_value_tracker")
        # pylint: enable=protected-access

        # Additional attributes specific to some quantized operators
        for attribute in metadata:
            if hasattr(obj, attribute):
                setattr(obj, attribute, metadata[attribute])

        return obj

    def dumps(self) -> str:
        """Dump itself to a string.

        Returns:
            metadata (str): String of the serialized object.
        """
        return dumps(self)

    def dump(self, file: TextIO) -> None:
        """Dump itself to a file.

        Args:
            file (TextIO): The file to dump the serialized object into.
        """
        dump(self, file)

    @classmethod
    def op_type(cls):
        """Get the type of this operation.

        Returns:
            op_type (str): The type of this operation, in the ONNX referential
        """
        return cls._impl_for_op_named

    @property
    def int_input_names(self):
        """Get the names of encrypted integer tensors that are used by this op.

        Returns:
            Set[str]: the names of the tensors

        """

        return self._int_input_names

    # Register node to our internal categories
    def __init_subclass__(cls, is_utility=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if is_utility:
            return
        ALL_QUANTIZED_OPS.add(cls)
        op_name = cls._impl_for_op_named
        if (op_name) is not None:
            ONNX_OPS_TO_QUANTIZED_IMPL[op_name] = cls
            candidate_impl = ONNX_OPS_TO_NUMPY_IMPL.get(op_name, None)
            cls.impl = candidate_impl if candidate_impl is not None else cls.impl

        assert_true(cls.impl is not None, f"Missing 'impl' for class {cls.__name__}")

        cls._populate_op_input_infos()
        cls._has_attr = len(cls._authorized_attr_names) > 0

    def __call__(self, *q_inputs: ONNXOpInputOutputType) -> ONNXOpInputOutputType:
        """Process the forward pass of the quantized op according to the implementation.

        The calibrate method needs to be called with sample data before using this function.

        Args:
            *q_inputs (ONNXOpInputOutputType): Quantized inputs.

        Returns:
            ONNXOpInputOutputType: Quantized output.
        """

        return self.q_impl(*q_inputs, **self.attrs)

    @classmethod
    def _populate_op_input_infos(cls):
        # for mypy
        assert cls.impl is not None
        if isinstance(cls.impl, ONNXMixedFunction):
            impl_signature = signature(cls.impl.function)
            cls._inputs_not_quantized = cls.impl.non_quant_params
            cls.produces_raw_output = cls.impl.output_is_raw
        else:
            impl_signature = signature(cls.impl)
        impl_params = impl_signature.parameters
        cls._params_name_to_input_idx = {val: index for index, val in enumerate(impl_params)}
        cls._input_idx_to_params_name = dict(enumerate(impl_params))
        cls._params_that_are_onnx_inputs = {
            param.name
            for param in impl_params.values()
            if param.kind in cls.POSITIONAL_ARGUMENTS_KINDS
        }
        cls._params_that_are_onnx_var_inputs = {
            param.name
            for param in impl_params.values()
            if param.kind in cls.VAR_POSITIONAL_ARGUMENTS_KINDS
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

    @classmethod
    def must_quantize_input(cls, input_name_or_idx: int) -> bool:
        """Determine if an input must be quantized.

        Quantized ops and numpy onnx ops take inputs and attributes. Inputs can be either constant
        or variable (encrypted). Note that this does not handle attributes, which are handled
        by QuantizedOp classes separately in their constructor.

        Args:
            input_name_or_idx (int): Index of the input to check.

        Returns:
            result (bool): Whether the input must be quantized (must be a `QuantizedArray`) or
                if it stays as a raw `numpy.array` read from ONNX.
        """

        # Operation parameters have names and indices, we only support indices here
        assert_true(isinstance(input_name_or_idx, int))
        input_name = cls._input_idx_to_params_name[input_name_or_idx]
        return input_name not in cls._inputs_not_quantized

    def q_impl(
        self,
        *q_inputs: ONNXOpInputOutputType,
        **attrs,
    ) -> ONNXOpInputOutputType:
        """Execute the quantized forward.

        Args:
            *q_inputs (ONNXOpInputOutputType): Quantized inputs.
            **attrs: the QuantizedOp attributes.

        Returns:
            ONNXOpInputOutputType: The returned quantized value.
        """

        # Here we need the float32 values from the QuantizedArrays. By default, when possible,
        # we want QuantizedOps to convert to TLUs. Ops that need to do quantized computation
        # will call _prepare_inputs_with_constants with quantize_actual_values=True
        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=False
        )
        f_outputs = self.call_impl(*prepared_inputs, **attrs)

        # If the op takes only raw values as inputs it must be producing only raw outputs
        # Operations such as Add/Mul can, in some settings, operate in this setting
        # for example in subgraphs that manipulate shapes
        if all(isinstance(q_input, RawOpOutput) for q_input in q_inputs):
            return f_outputs.view(RawOpOutput)

        return self.prepare_output(f_outputs)

    def _prepare_constants(
        self, number_of_inputs: int, calibrate: bool, quantize_actual_values: bool
    ) -> List[Optional[ONNXOpInputOutputType]]:
        """Prepare a list of inputs of the quantized op, filling in the constants.

        Args:
            number_of_inputs (int): The total number of inputs to fill in
            calibrate (bool): A flag specifying if the method is called during calibration
            quantize_actual_values (bool): If called by a quantized operator that does matrix
                multiplication between encrypted and clear values, this method will apply
                the quantization computation to the input, which will be fused in a (potentially
                larger) TLU, with preceding floating point computations

        Returns:
            result (List): a list of inputs which are either QuantizedArray or numpy.arrays. If
                quantize_actual_values==True then the constants are assumed to be quantized
        """

        prepared_inputs: List[Optional[ONNXOpInputOutputType]]

        prepared_inputs = [None] * number_of_inputs

        # If calibrating (calibrate=True): inputs are numpy.ndarrays of float32
        # If used in the computation graph (calibrate=False): inputs are QuantizedArrays, of which
        # we use the float32 .values and optionally we quantized them to int

        # Constants are quantized during graph creation. If calibrating, pass through
        # the original float values. If running in the computation graph, if the quantized values
        # are requested, use the QuantizedArray object. Finally, if running the graph and
        # the original float values are requested, then return the QuantizedArray.values
        for input_idx, constant_val in self.constant_inputs.items():
            # If calibrating or if a float input is required (for TLU fusion)
            # an input that **must** be quantized will be stored in a QuantizedArray, we need
            # to retrieve its .values to return the requested float values

            # In all other cases we return a QuantizedArray or numpy.array. QuantizedArrays
            # in this case are produced by other ops or inputs. numpy.arrays are produced
            # by initializers
            if calibrate or not quantize_actual_values:
                is_clear_value = isinstance(constant_val, RawOpOutput)

                if not is_clear_value and self.__class__.must_quantize_input(input_idx):
                    prepared_inputs[input_idx] = constant_val.values
                else:
                    prepared_inputs[input_idx] = constant_val
            else:
                prepared_inputs[input_idx] = constant_val

        return prepared_inputs

    def _prepare_quantized_input(self, input_: QuantizedArray) -> QuantizedArray:
        """Prepare a quantized encrypted input for a univariate or mixin operation.

        Args:
            input_ (QuantizedArray): the encrypted input

        Returns:
            result (QuantizedArray): the quantized input, either re-quantized to new quantization
                options, or keeping the original quantization
        """

        # Here we want to trace the code that does quantization, to produce a TLU
        # We use the op's input quantization options

        quant_opts = QuantizationOptions(self.input_quant_opts.n_bits)
        quant_opts.copy_opts(self.input_quant_opts)

        assert_false(
            self.can_fuse(),
            f"The {self.__class__.__name__} operation is attempting "
            "to quantize its inputs but is marked as fusable (can_fuse() return True). ",
        )

        # Now we quantize the input. For ops that require quantized inputs (which
        # call this function with quantize_actual_values==True), this
        # will produce numpy ops that will be fused to a TLU. Fusable ops will
        # use the .values directly (see else branch below). We need the quantization
        # stats to generate the quantization code, they cannot be computed on the fly.

        # Conv/Matmul layers have quantization options initialized by the PTQ default,
        # but when parsing the ONNX graph, some options can be overwritten. Thus
        # when evaluating QAT layers we ignore one of these options to allow the
        # override.
        if quant_opts.is_equal(input_.quantizer.quant_options, ignore_sign_qat=True):
            # Pass-through the input quantizer when the input is already quantized in
            # the manner that this op requires: this makes the op use the qvalues directly,
            # in q_impl and will avoid a TLU to re-quantize.
            new_input = QuantizedArray(
                quant_opts.n_bits,
                input_.qvalues,
                value_is_float=False,
                options=input_.quantizer.quant_options,
                stats=input_.quantizer.quant_stats,
                params=input_.quantizer.quant_params,
            )
        else:
            quant_params = None
            if input_.quantizer.is_qat:
                quant_params = input_.quantizer.quant_params

            new_input = QuantizedArray(
                quant_opts.n_bits,
                input_.values,
                options=quant_opts,
                stats=input_.quantizer.quant_stats,
                params=quant_params,
            )

        return new_input

    def _prepare_inputs_with_constants(
        self,
        *inputs: ONNXOpInputOutputType,
        calibrate: bool,
        quantize_actual_values: bool,
    ):
        """Retrieve all the inputs of an operator in the computational graph.

        This helper method will prepare a list of inputs to an operator. Inputs can be variables,
        i.e., encrypted tensors, or constants (in the clear). Inputs to an operator are set-up in
        the slots of a list, as the order of inputs is important.

        Usually the input list is populated with QuantizedArrays. Operators that require the
        original float (operators that only produce or contribute to TLUs) values will just read
        out the .values of  these quantized arrays. Operators that do matrix multiplication will
        read out the quantized integer values of the arrays.  The method can be called during
        calibration, in which case the variable inputs are just float numpy tensors.

        Args:
             *inputs (ONNXOpInputOutputType): A list of all variable inputs
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
        num_inputs = len(inputs)
        is_param_variadic = len(self._params_that_are_onnx_var_inputs) > 0

        condition_inputs = (
            num_onnx_inputs >= (num_inputs) + num_provided_constants >= num_required_onnx_inputs
            if not is_param_variadic
            else True
        )
        assert_true(
            condition_inputs,
            f"This operator has {num_onnx_inputs} ONNX inputs, and {num_provided_constants} "
            "constants were already provided when instantiating the class. "
            f"Got a call with {num_inputs} inputs and constants while the call expects between "
            f"{num_required_onnx_inputs} and {num_onnx_inputs} inputs and constants.",
        )

        prepared_inputs = self._prepare_constants(
            num_inputs if is_param_variadic else num_onnx_inputs,
            calibrate,
            quantize_actual_values,
        )

        # If calibrating, the function is called with numpy.ndarrays
        # If not calibrating, the function is called with QuantizedArray inputs
        # If quantized values are requested, we quantized the float32 values contained in the
        # QuantizedArrays, else we return the float32 values directly.

        curr_input_fill_idx = 0
        for input_idx, input_ in enumerate(inputs):
            while prepared_inputs[curr_input_fill_idx] is not None:
                curr_input_fill_idx += 1

            # This is an integer scalar (e.g., tensor shape). This is not an encrypted
            # value, it is not traced
            is_clear_value = isinstance(input_, RawOpOutput)

            if input_ is None:
                # Do nothing if the input is not set, the underlying ops will handle the None
                pass
            elif calibrate or is_clear_value:
                # This is used during calibration with numpy.ndarrays
                # or then the input is raw (not quantized)
                prepared_inputs[curr_input_fill_idx] = input_
            elif quantize_actual_values:
                # This is used by mixing (conv/gemm) or value re-arranging ops (reshape)
                input_ = cast(QuantizedArray, input_)
                new_input = self._prepare_quantized_input(input_)

                # Check that the input quantizer is correct - that it can de-quantize
                # values correctly. If it is not, it is added to the list of invalid tensors
                # for which an error is raised
                if (
                    new_input.quantizer.is_qat
                    and not input_.quantizer.is_precomputed_qat
                    and self.error_tracker is not None
                    and not new_input.quantizer.check_is_uniform_quantized(
                        new_input.quantizer.quant_options
                    )
                ):
                    self.error_tracker.append(input_idx)

                prepared_inputs[curr_input_fill_idx] = new_input
            else:
                # This is usually used for univariate ops that are fused into TLUs
                input_ = cast(QuantizedArray, input_)
                prepared_inputs[curr_input_fill_idx] = input_.values

            curr_input_fill_idx += 1

        if self.debug_value_tracker is not None:
            # For mypy
            assert self.debug_value_tracker is not None
            assert self.op_instance_name is not None

            # pylint: disable-next=unsupported-assignment-operation
            self.debug_value_tracker[self.op_instance_name] = {
                k: prepared_inputs[k] for k in range(len(prepared_inputs))
            }

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

        raw_result = self.call_impl(*prepared_inputs, **self.attrs)
        if isinstance(raw_result, RawOpOutput):
            return raw_result

        quantized_samples = QuantizedArray(self.n_bits, raw_result)

        self.output_quant_params = quantized_samples.quantizer.quant_params
        self.output_quant_stats = quantized_samples.quantizer.quant_stats

        return quantized_samples.values

    def prepare_output(self, qoutput_activation: numpy.ndarray) -> QuantizedArray:
        """Quantize the output of the activation function.

        The calibrate method needs to be called with sample data before using this function.

        Args:
            qoutput_activation (numpy.ndarray): Output of the activation function.

        Returns:
            QuantizedArray: Quantized output.
        """

        assert_true(
            self.output_quant_params is not None,
            f"output quantization params was None for class {self.__class__.__name__}, "
            "did you forget to call calibrate with sample data?",
        )

        return QuantizedArray(
            self.n_bits,
            qoutput_activation,
            value_is_float=True,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=self.output_quant_params,
        )

    def call_impl(
        self, *inputs: Union[numpy.ndarray, QuantizedArray, None], **attrs
    ) -> numpy.ndarray:
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
        if isinstance(self.impl, ONNXMixedFunction):
            impl_func = self.impl.function
        else:
            impl_func = self.impl.__func__  # type: ignore
        outputs = impl_func(*inputs) if not self._has_attr else impl_func(*inputs, **attrs)
        assert_true(
            isinstance(outputs, tuple),
            f"The output of {impl_func.__name__} needs to be a tuple. Got {outputs}",
        )
        num_outputs = len(outputs)
        assert_true(
            (num_outputs) == 1,
            f"Currently only single output ops are supported, got {num_outputs} outputs.",
        )

        return outputs[0]

    def can_fuse(self) -> bool:  # pylint: disable=no-self-use
        """Determine if the operator impedes graph fusion.

        This function shall be overloaded by inheriting classes to test self._int_input_names, to
        determine whether the operation can be fused to a TLU or not. For example an operation
        that takes inputs produced by a unique integer tensor can be fused to a TLU. Example:
        f(x) = x * (x + 1) can be fused. A function that does f(x) = x * (x @ w + 1) can't be fused.

        Returns:
            bool: whether this QuantizedOp instance produces Concrete code that can be fused to TLUs
        """
        return True

    def _get_output_quant_opts(self):
        """Return the output quantization options.

        This function computes the output quantization options based on the input quantizer
        and whether the operation can be fused to a TLU.

        Returns:
            output_quant_opts (QuantizationOptions): the options for output quantization
        """

        output_quant_opts = QuantizationOptions(self.input_quant_opts.n_bits)
        output_quant_opts.copy_opts(self.input_quant_opts)
        if self.can_fuse():
            output_quant_opts.is_qat = False
        return output_quant_opts


class QuantizedOpUnivariateOfEncrypted(QuantizedOp, is_utility=True):
    """An univariate operator of an encrypted value.

    This operation is not really operating as a quantized operation. It is useful when the
    computations get fused into a TLU, as in e.g., Act(x) = x || (x + 42)).
    """

    def __init__(
        self,
        n_bits_output: int,
        op_instance_name: str,
        int_input_names: Optional[Set[str]] = None,
        constant_inputs: Optional[Union[Dict[str, Any], Dict[int, Any]]] = None,
        input_quant_opts: Optional[QuantizationOptions] = None,
        **attrs,
    ) -> None:
        # Disable mypy which is failing for mixins
        super().__init__(
            n_bits_output,
            op_instance_name,
            int_input_names,
            constant_inputs,
            input_quant_opts,
            **attrs,
        )  # type: ignore

        # We do not support this type of operation between encrypted tensors, only between:
        # - encrypted tensors and float constants
        # - tensors that are produced by a unique integer tensor
        # If this operation is applied between two constants
        # it should be optimized out by the constant folding procedure
        assert_true(
            self.can_fuse() or (constant_inputs is not None and len(constant_inputs) == 1),
            "Do not support this type of operation between encrypted tensors",
        )

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        This operation can be fused and computed in float when a single integer tensor generates
        both the operands. For example in the formula: f(x) = x || (x + 1)  where x is an integer
        tensor.

        Returns:
            bool: Can fuse
        """

        # for mypy
        assert isinstance(self, QuantizedOp)

        return len(self._int_input_names) == 1  # pylint: disable=no-member


class QuantizedMixingOp(QuantizedOp, is_utility=True):
    """An operator that mixes (adds or multiplies) together encrypted inputs.

    Mixing operators cannot be fused to TLUs.
    """

    lsbs_to_remove: Optional[int] = None
    rounding_threshold_bits: Optional[int] = None

    def __init__(self, *args, rounding_threshold_bits: Optional[int] = None, **kwargs) -> None:
        """Initialize quantized ops parameters plus specific parameters.

        Args:
            rounding_threshold_bits (Optional[int]): Number of bits to round to.
            *args: positional argument to pass to the parent class.
            **kwargs: named argument to pass to the parent class.
        """
        self.rounding_threshold_bits = rounding_threshold_bits
        super().__init__(*args, **kwargs)

    def can_fuse(self) -> bool:
        """Determine if this op can be fused.

        Mixing operations cannot be fused since it must be performed over integer tensors and it
        combines different encrypted elements of the input tensors. Mixing operations are Conv,
        MatMul, etc.

        Returns:
            bool: False, this operation cannot be fused as it adds different encrypted integers
        """

        return False

    def make_output_quant_parameters(
        self,
        q_values: Union[numpy.ndarray, Any],
        scale: numpy.float64,
        zero_point: Union[int, float, numpy.ndarray],
    ) -> QuantizedArray:
        """Build a quantized array from quantized integer results of the op and quantization params.

        Args:
            q_values (Union[numpy.ndarray, Any]): the quantized integer values to wrap
                in the QuantizedArray
            scale (float): the pre-computed scale of the quantized values
            zero_point (Union[int, float, numpy.ndarray]): the pre-computed zero_point of
                the q_values

        Returns:
            QuantizedArray: the quantized array that will be passed to the QuantizedModule output.
        """

        out_opts = self._get_output_quant_opts()
        out_opts.is_signed = False
        out_opts.is_symmetric = False

        # Since we don't know the real bit-width of these quantized values,
        # return a quantizer that has zero offset
        out_params = UniformQuantizationParameters(
            scale=scale,
            zero_point=zero_point,
            offset=0,
        )

        return QuantizedArray(
            self.n_bits,
            q_values,
            value_is_float=False,
            options=self._get_output_quant_opts(),
            stats=self.output_quant_stats,
            params=out_params,
        )

    def cnp_round(
        self, x: Union[numpy.ndarray, fhe.tracing.Tracer], calibrate_rounding: bool
    ) -> numpy.ndarray:
        """Round the input array to the specified number of bits.

        Args:
            x (Union[numpy.ndarray, fhe.tracing.Tracer]): The input array to be rounded.
            calibrate_rounding (bool): Whether to calibrate the rounding
                (compute the lsbs_to_remove)

        Returns:
            numpy.ndarray: The rounded array.
        """

        # Rounding is applied only if specified by user
        if self.rounding_threshold_bits is not None:
            if calibrate_rounding:
                assert_true(
                    not isinstance(x, fhe.tracing.Tracer),
                    "Can't compute lsbs_to_remove at compilation time.",
                )
                assert_true(
                    self.lsbs_to_remove is None,
                    "Rounding has already been calibrated.",
                )

                current_n_bits_accumulator = compute_bits_precision(x)
                self.lsbs_to_remove = current_n_bits_accumulator - self.rounding_threshold_bits

            # mypy
            assert self.lsbs_to_remove is not None

            # Apply rounding if needed
            if self.lsbs_to_remove > 0:
                x = fhe.round_bit_pattern(x, lsbs_to_remove=self.lsbs_to_remove)

        return x
