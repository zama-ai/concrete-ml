"""Quantization utilities for a numpy array/tensor."""
# pylint: disable=too-many-lines
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, TextIO, Union, get_type_hints

import numpy

from ..common.debugging import assert_true
from ..common.serialization.dumpers import dump, dumps

STABILITY_CONST = 10**-6


def fill_from_kwargs(obj, klass, **kwargs):
    """Fill a parameter set structure from kwargs parameters.

    Args:
        obj: an object of type klass, if None the object is created if any of the type's
            members appear in the kwargs
        klass: the type of object to fill
        kwargs: parameter names and values to fill into an instance of the klass type

    Returns:
        obj: an object of type klass
        kwargs: remaining parameter names and values that were not filled into obj

    Raises:
        TypeError: if the types of the parameters in kwargs could not be converted
            to the corresponding types of members of klass
    """

    # Get the members of the parameter set structure
    hints = get_type_hints(klass)

    # Keep track of the parameters that were used
    list_args_used = []
    for name, value in kwargs.items():
        # If this parameter is not in the structure, ignore it
        if name not in hints:
            continue

        # Create the structure if needed, otherwise continue filling an existing structure
        if obj is None:
            obj = klass()

        # Set the parameter in the structure
        setattr(obj, name, value)

        # Mark the parameter as used
        list_args_used.append(name)

    # Remove all the parameters that were filled in the structure for kwargs
    # Keep all other kwargs parameters to be able to fill other structures
    for name in list_args_used:
        kwargs.pop(name)

    # If the structure was created or modified by a call to this function, check
    # that it is completely filled
    if obj is not None:
        for name in hints:
            if getattr(obj, name) is None:
                raise TypeError(f"Missing quantizer parameter {name}")

    # Return the parameter structure and the kwargs with the used parameters removed
    return obj, kwargs


class QuantizationOptions:
    """Options for quantization.

    Determines the number of bits for quantization and the method of quantization of the values.
    Signed quantization allows negative quantized values. Symmetric quantization assumes the float
    values are distributed symmetrically around x=0 and assigns signed values around 0 to the float
    values. QAT (quantization aware training) quantization assumes the values are already quantized,
    taking a discrete set of values, and assigns these values to integers, computing only the scale.
    """

    n_bits: int

    # Determines whether the integer value representing the quantized floats are signed
    is_signed: bool = False

    # Determines whether to use symmetric quantization and/or whether the values have a
    # symmetric distribution with respect to 0
    is_symmetric: bool = False

    # Determines whether the values handled by the Quantizer are already quantized through
    # quantization aware training
    is_qat: bool = False

    # Determine if the quantized integer values should only be in [2^(n-1)+1 .. 2^(n-1)-1]
    # or whether we use the full range. Only implemented for QAT
    is_narrow: bool = False

    # Determines whether the values handled by the quantizer were produced by a custom
    # quantization layer that has pre-computed scale and zero-point
    #   (i.e., ONNX brevitas quant layer)
    is_precomputed_qat: bool = False

    def __init__(
        self, n_bits: int, is_signed: bool = False, is_symmetric: bool = False, is_qat: bool = False
    ):
        self.n_bits = n_bits
        self.is_signed = is_signed
        self.is_symmetric = is_symmetric
        self.is_qat = is_qat

        # QAT quantization is not symmetric
        assert_true(not self.is_qat or not self.is_symmetric)

        # Symmetric quantization is signed
        assert_true(not self.is_symmetric or self.is_signed)

    def __eq__(self, other) -> bool:
        return (
            other.n_bits == self.n_bits
            and other.is_signed == self.is_signed
            and other.is_symmetric == self.is_symmetric
            and other.is_qat == self.is_qat
            and other.is_narrow == self.is_narrow
            and other.is_precomputed_qat == self.is_precomputed_qat
        )

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        metadata["n_bits"] = self.n_bits
        metadata["is_signed"] = self.is_signed
        metadata["is_symmetric"] = self.is_symmetric
        metadata["is_qat"] = self.is_qat
        metadata["is_narrow"] = self.is_narrow
        metadata["is_precomputed_qat"] = self.is_precomputed_qat
        return metadata

    @staticmethod
    def load_dict(metadata: Dict):
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            QuantizationOptions: The loaded object.
        """
        obj = QuantizationOptions(
            n_bits=metadata["n_bits"],
            is_symmetric=metadata["is_symmetric"],
            is_signed=metadata["is_signed"],
            is_qat=metadata["is_qat"],
        )
        for attr_name in ["is_narrow", "is_precomputed_qat"]:
            setattr(obj, attr_name, metadata[attr_name])
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

    def copy_opts(self, opts):
        """Copy the options from a different structure.

        Args:
            opts (QuantizationOptions): structure to copy parameters from.
        """

        self.n_bits = opts.n_bits
        self.is_signed = opts.is_signed
        self.is_symmetric = opts.is_symmetric
        self.is_qat = opts.is_qat
        self.is_precomputed_qat = opts.is_precomputed_qat
        self.is_narrow = opts.is_narrow

    @property
    def quant_options(self):
        """Get a copy of the quantization parameters.

        Returns:
            UniformQuantizationParameters: a copy of the current quantization parameters
        """

        # Note we don't deepcopy here, since this is a mixin
        res = QuantizationOptions(self.n_bits)
        res.copy_opts(self)
        return res

    def is_equal(self, opts, ignore_sign_qat: bool = False) -> bool:
        """Compare two quantization options sets.

        Args:
            opts (QuantizationOptions): options to compare this instance to
            ignore_sign_qat (bool): ignore sign comparison for QAT options

        Returns:
            bool: whether the two quantization options compared are equivalent
        """
        sign_equals = True if opts.is_qat and ignore_sign_qat else opts.is_signed == self.is_signed
        return (
            opts.is_qat == self.is_qat
            and sign_equals
            and opts.is_symmetric == self.is_symmetric
            and opts.n_bits == self.n_bits
        )


class MinMaxQuantizationStats:
    """Calibration set statistics.

    This class stores the statistics for the calibration set or for a calibration data batch.
    Currently we only store min/max to determine the quantization range. The min/max are computed
    from the calibration set.
    """

    rmax: Optional[float] = None
    rmin: Optional[float] = None
    uvalues: Optional[numpy.ndarray] = None

    def __init__(
        self,
        rmax: Optional[float] = None,
        rmin: Optional[float] = None,
        uvalues: Optional[numpy.ndarray] = None,
    ):
        self.rmax = rmax
        self.rmin = rmin
        self.uvalues = uvalues

    def __eq__(self, other) -> bool:
        # Disable mypy as numpy.array_equal properly handles None types
        return (
            other.rmax == self.rmax
            and other.rmin == self.rmin
            and numpy.array_equal(other.uvalues, self.uvalues)  # type: ignore[arg-type]
        )

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        metadata["rmax"] = self.rmax
        metadata["rmin"] = self.rmin
        metadata["uvalues"] = self.uvalues
        return metadata

    @staticmethod
    def load_dict(metadata: Dict):
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            QuantizationOptions: The loaded object.
        """
        to_return = MinMaxQuantizationStats(
            rmax=metadata["rmax"],
            rmin=metadata["rmin"],
            uvalues=metadata["uvalues"],
        )

        return to_return

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

    def compute_quantization_stats(self, values: numpy.ndarray) -> None:
        """Compute the calibration set quantization statistics.

        Args:
            values (numpy.ndarray): Calibration set on which to compute statistics.
        """

        self.rmin = numpy.min(values)
        self.rmax = numpy.max(values)

        # To find unique float values we need to round. We round to 2 decimal figures.
        # Floating point inaccuracies in computation can lead to differences in the last
        # decimal figures. We want to ignore such differences but also avoid
        # coalescing float values that should be distinct
        rvalues = numpy.round(values, decimals=2)

        # Unique values from the distribution sample. These values are sorted
        # in order to extract the quantization scale in the case of QAT
        self.uvalues = numpy.unique(rvalues)

    @property
    def quant_stats(self):
        """Get a copy of the calibration set statistics.

        Returns:
            MinMaxQuantizationStats: a copy of the current quantization stats
        """

        # Note we don't deepcopy here, since this is a mixin
        res = MinMaxQuantizationStats()
        res.copy_stats(self)
        return res

    def copy_stats(self, stats) -> None:
        """Copy the statistics from a different structure.

        Args:
            stats (MinMaxQuantizationStats): structure to copy statistics from.
        """

        self.rmax = stats.rmax
        self.rmin = stats.rmin
        self.uvalues = stats.uvalues

    def check_is_uniform_quantized(self, options: QuantizationOptions) -> bool:
        """Check if these statistics correspond to uniformly quantized values.

        Determines whether the values represented by this QuantizedArray show
        a quantized structure that allows to infer the scale of quantization.

        Args:
            options (QuantizationOptions): used to quantize the values in the QuantizedArray

        Returns:
            bool: check result.
        """

        assert self.uvalues is not None

        if self.uvalues.size > 2**options.n_bits:
            return False

        if self.uvalues.size == 1:
            return False

        unique_scales = numpy.unique(numpy.diff(self.uvalues))
        min_scale = unique_scales[0]

        re_quant_scales = numpy.rint(unique_scales / min_scale) * min_scale

        return numpy.allclose(unique_scales, re_quant_scales, atol=0.02)


class UniformQuantizationParameters:
    """Quantization parameters for uniform quantization.

    This class stores the parameters used for quantizing real values to discrete integer values.
    The parameters are computed from quantization options and quantization statistics.
    """

    scale: Optional[numpy.float64] = None
    zero_point: Optional[Union[int, float, numpy.ndarray]] = None
    offset: Optional[int] = None

    def __init__(
        self,
        scale: Optional[numpy.float64] = None,
        zero_point: Optional[Union[int, float, numpy.ndarray]] = None,
        offset: Optional[int] = None,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.offset = offset

    def __eq__(self, other) -> bool:
        return (
            other.scale == self.scale
            and other.zero_point == self.zero_point
            and other.offset == self.offset
        )

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        metadata["scale"] = self.scale
        metadata["zero_point"] = self.zero_point
        metadata["offset"] = self.offset
        return metadata

    @staticmethod
    def load_dict(metadata: Dict) -> UniformQuantizationParameters:
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            UniformQuantizationParameters: The loaded object.
        """
        to_return = UniformQuantizationParameters(
            scale=metadata["scale"],
            zero_point=metadata["zero_point"],
            offset=metadata["offset"],
        )
        return to_return

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

    def copy_params(self, params) -> None:
        """Copy the parameters from a different structure.

        Args:
            params (UniformQuantizationParameters): parameter structure to copy
        """

        self.scale = params.scale
        self.zero_point = params.zero_point
        self.offset = params.offset

    @property
    def quant_params(self):
        """Get a copy of the quantization parameters.

        Returns:
            UniformQuantizationParameters: a copy of the current quantization parameters
        """

        # Note we don't deepcopy here, since this is a mixin
        res = UniformQuantizationParameters()
        res.copy_params(self)
        return res

    def compute_quantization_parameters(
        self, options: QuantizationOptions, stats: MinMaxQuantizationStats
    ) -> None:
        """Compute the quantization parameters.

        Args:
            options (QuantizationOptions): quantization options set
            stats (MinMaxQuantizationStats): calibrated statistics for quantization
        """

        self.offset = 0
        if options.is_signed:
            self.offset = 2 ** (options.n_bits - 1)

        assert_true(not options.is_symmetric or options.is_signed)
        assert_true(not options.is_qat or not options.is_symmetric)

        # for mypy
        assert stats.rmax is not None
        assert stats.rmin is not None

        # Small constant needed for stability
        if stats.rmax - stats.rmin < STABILITY_CONST:
            # In this case there is  a single unique value to quantize

            # is is_signed is True, we need to set the offset back to 0.
            # Signed quantization does not make sense for a single value.
            self.offset = 0

            # This value could be multiplied with inputs at some point in the model
            # Since zero points need to be integers, if this value is a small float (ex: 0.01)
            # it will be quantized to 0 with a 0 zero-point, thus becoming useless in multiplication

            if numpy.abs(stats.rmax) < STABILITY_CONST:
                # If the value is a 0 we cannot do it since the scale would become 0 as well
                # resulting in division by 0
                self.scale = numpy.float64(1.0)
                # Ideally we should get rid of round here but it is risky
                # regarding the FHE compilation.
                # Indeed, the zero_point value for the weights has to be an integer
                # for the compilation to work.
                self.zero_point = numpy.rint(-stats.rmin).astype(numpy.int64)
            else:
                # If the value is not a 0 we can tweak the scale factor so that
                # the value quantizes to 1
                self.scale = numpy.float64(stats.rmax)
                self.zero_point = 0
        else:
            if options.is_symmetric:
                assert_true(not options.is_qat)
                self.zero_point = 0
                self.scale = (
                    numpy.maximum(numpy.abs(stats.rmax), numpy.abs(stats.rmin))
                    / ((2**options.n_bits - 1 - self.offset))
                ).astype(numpy.float64)
            else:
                # Infer the QAT parameters if this is a custom QAT network
                # which does not store scale/zero-point in the ONNX directly.

                # Do not infer the parameters if the network was trained with Brevitas
                # they are stored in the ONNX file and are the true quantization parameters
                # used in training - no need to infer them.

                # If the parameters do not appear quantized, use PTQ for quantization.
                # The QuantizedModule will perform error checking of quantized tensors
                # and will issue an error if the network is not well quantized during training
                if (
                    options.is_qat
                    and not options.is_precomputed_qat
                    and stats.uvalues is not None
                    and stats.check_is_uniform_quantized(options)
                ):
                    assert_true(
                        len(stats.uvalues) > 1,
                        "A single unique value was detected in a tensor of "
                        "quantized values in a QAT import.\n"
                        "Please check the stability thresholds.\n"
                        "This can occur with a badly trained model.",
                    )
                    unique_scales = numpy.unique(numpy.diff(stats.uvalues))
                    self.scale = numpy.float64(unique_scales[0])

                if self.scale is None:
                    self.scale = numpy.float64(
                        (stats.rmax - stats.rmin) / (2**options.n_bits - 1)
                        if stats.rmax != stats.rmin
                        else 1.0
                    )

                if options.is_qat:
                    self.zero_point = 0
                else:
                    # for mypy
                    assert self.offset is not None

                    self.zero_point = numpy.round(
                        (
                            # Pylint does not see that offset is not None here
                            # pylint: disable-next=invalid-unary-operand-type
                            stats.rmax * (-self.offset)
                            - (stats.rmin * (2**options.n_bits - 1 - self.offset))
                        )
                        / (stats.rmax - stats.rmin)
                    ).astype(numpy.int64)


# Change UniformQuantizer inheritance from UniformQuantizationParameters to composition.
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1434
class UniformQuantizer(UniformQuantizationParameters, QuantizationOptions, MinMaxQuantizationStats):
    """Uniform quantizer.

    Contains all information necessary for uniform quantization and provides
    quantization/de-quantization functionality on numpy arrays.

    Args:
        options (QuantizationOptions): Quantization options set
        stats (Optional[MinMaxQuantizationStats]): Quantization batch statistics set
        params (Optional[UniformQuantizationParameters]): Quantization parameters set
            (scale, zero-point)
    """

    # pylint: disable-next=super-init-not-called
    def __init__(
        self,
        options: Optional[QuantizationOptions] = None,
        stats: Optional[MinMaxQuantizationStats] = None,
        params: Optional[UniformQuantizationParameters] = None,
        no_clipping: bool = False,
    ):
        if options is not None:
            self.copy_opts(options)

        if stats is not None:
            self.copy_stats(stats)

        if params is not None:
            self.copy_params(params)

        self.no_clipping = no_clipping

        # Force scale to be a float64
        if self.scale is not None:
            self.scale = numpy.float64(self.scale)

    def __eq__(self, other) -> bool:

        is_equal = other.no_clipping == self.no_clipping

        for attribute in [
            "n_bits",
            "is_signed",
            "is_symmetric",
            "is_qat",
            "is_narrow",
            "is_precomputed_qat",
            "rmax",
            "rmin",
            "uvalues",
            "scale",
            "zero_point",
            "offset",
        ]:
            # All possible attributes in a UniformQuantizer object are not necessarily available
            value_1, value_2 = getattr(other, attribute, None), getattr(self, attribute, None)

            # If the first value is a numpy array, check that the second value is also a numpy array
            # and that values are all equal
            if isinstance(value_1, numpy.ndarray):
                is_equal &= isinstance(value_2, numpy.ndarray) and numpy.array_equal(
                    value_1, value_2
                )

            # Else, check that both values are equal. Here, we expect both values to have a __eq__
            # operator implemented and which returns a boolean
            else:
                is_equal &= value_1 == value_2

        return is_equal

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        for attribute in [
            "n_bits",
            "is_signed",
            "is_symmetric",
            "is_qat",
            "is_narrow",
            "is_precomputed_qat",
            "rmax",
            "rmin",
            "uvalues",
            "scale",
            "zero_point",
            "offset",
            "no_clipping",
        ]:
            if hasattr(self, attribute):
                metadata[attribute] = getattr(self, attribute)

        return metadata

    @staticmethod
    def load_dict(metadata: Dict) -> UniformQuantizer:
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            UniformQuantizer: The loaded object.
        """

        # Instantiate the quantizer
        obj = UniformQuantizer()

        for attribute in [
            "n_bits",
            "is_signed",
            "is_symmetric",
            "is_qat",
            "is_narrow",
            "is_precomputed_qat",
            "rmax",
            "rmin",
            "uvalues",
            "scale",
            "zero_point",
            "offset",
            "no_clipping",
        ]:
            if attribute in metadata:
                setattr(obj, attribute, metadata[attribute])

        # The `uvalues` attribute needs to be put back to a numpy.array object
        if "uvalues" in metadata:
            obj.uvalues = metadata["uvalues"]

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

    def quant(self, values: numpy.ndarray) -> numpy.ndarray:
        """Quantize values.

        Args:
            values (numpy.ndarray): float values to quantize

        Returns:
            numpy.ndarray: Integer quantized values.
        """

        # for mypy
        assert self.zero_point is not None
        assert self.offset is not None
        assert self.scale is not None

        qvalues = numpy.rint(values / self.scale + self.zero_point)

        # Clipping can be performed for PTQ and for precomputed (for now only Brevitas) QAT
        # (where quantizer parameters are available in ONNX layers).
        # For Custom QAT, with inferred parameters the type of quantization (signed/narrow)
        # can not be inferred and thus clipping can not be performed reliably
        # It is possible to disable this clipping step for specific cases such as quantizing values
        # within fully-leveled circuits (where not bounds are needed)
        if (not self.is_qat or self.is_precomputed_qat) and not self.no_clipping:
            # Offset is either 2^(n-1) or 0, but for narrow range
            # the values should be clipped to [2^(n-1)+1, .. 2^(n-1)-1], so we add
            # one to the minimum value for narrow range
            # Pylint does not see that offset is not None here
            # pylint: disable-next=invalid-unary-operand-type
            min_value = -self.offset
            if self.is_narrow:
                min_value += 1

            qvalues = qvalues.clip(min_value, 2 ** (self.n_bits) - 1 - self.offset)

        return qvalues.astype(numpy.int64)

    def dequant(self, qvalues: numpy.ndarray) -> Union[Any, numpy.ndarray]:
        """De-quantize values.

        Args:
            qvalues (numpy.ndarray): integer values to de-quantize

        Returns:
            Union[Any, numpy.ndarray]: De-quantized float values.
        """

        # for mypy
        assert self.zero_point is not None
        assert self.scale is not None

        assert_true(
            isinstance(self.scale, (numpy.floating, float))
            or (isinstance(self.scale, numpy.ndarray) and self.scale.dtype is numpy.float64),
            "Scale is a of type "
            + type(self.scale).__name__
            + ((" " + str(self.scale.dtype)) if isinstance(self.scale, numpy.ndarray) else ""),
        )

        ans = self.scale * (qvalues - numpy.asarray(self.zero_point, dtype=numpy.float64))

        return ans


class QuantizedArray:
    """Abstraction of quantized array.

    Contains float values and their quantized integer counter-parts. Quantization is performed
    by the quantizer member object. Float and int values are kept in sync. Having both types
    of values is useful since quantized operators in Concrete ML graphs might need one or the other
    depending on how the operator works (in float or in int). Moreover, when the encrypted
    function needs to return a value, it must return integer values.

    See https://arxiv.org/abs/1712.05877.

    Args:
        values (numpy.ndarray): Values to be quantized.
        n_bits (int): The number of bits to use for quantization.
        value_is_float (bool, optional): Whether the passed values are real (float) values or not.
            If False, the values will be quantized according to the passed scale and zero_point.
            Defaults to True.
        options (QuantizationOptions): Quantization options set
        stats (Optional[MinMaxQuantizationStats]): Quantization batch statistics set
        params (Optional[UniformQuantizationParameters]): Quantization parameters set
            (scale, zero-point)
        kwargs: Any member of the options, stats, params sets as a key-value pair. The parameter
            sets need to be completely parametrized if their members appear in kwargs.
    """

    quantizer: UniformQuantizer
    values: numpy.ndarray
    qvalues: numpy.ndarray

    def __init__(
        self,
        n_bits,
        values: Optional[numpy.ndarray],
        value_is_float: bool = True,
        options: Optional[QuantizationOptions] = None,
        stats: Optional[MinMaxQuantizationStats] = None,
        params: Optional[UniformQuantizationParameters] = None,
        **kwargs,
    ):
        # If no options were passed, create a default options structure with the required n_bits
        options = deepcopy(options) if options is not None else QuantizationOptions(n_bits)

        # Override the options number of bits if an options structure was provided
        # with the number of bits specified by the caller.
        options.n_bits = n_bits
        self.n_bits = n_bits

        options, kwargs = fill_from_kwargs(options, QuantizationOptions, **kwargs)
        stats, kwargs = fill_from_kwargs(stats, MinMaxQuantizationStats, **kwargs)
        params, kwargs = fill_from_kwargs(params, UniformQuantizationParameters, **kwargs)

        # All kwargs should belong to one of the parameter sets, anything else is unsupported
        if len(kwargs) > 0:
            str_invalid_keywords = ",".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments '{str_invalid_keywords}'")

        # Create the quantizer from the provided parameter sets
        # Some parameters could be None and are computed below
        self.quantizer = UniformQuantizer(options, stats, params)

        if values is not None:
            self._values_setup(values, value_is_float, options, stats, params)

    def __eq__(self, other):
        is_equal = other.n_bits == self.n_bits and other.quantizer == self.quantizer

        for attribute in ["values", "qvalues"]:
            is_equal &= numpy.array_equal(getattr(other, attribute), getattr(self, attribute))

        return is_equal

    def _values_setup(
        self,
        values: numpy.ndarray,
        value_is_float: bool,
        options: QuantizationOptions = None,
        stats: Optional[MinMaxQuantizationStats] = None,
        params: Optional[UniformQuantizationParameters] = None,
    ):
        """Set up the values of the quantized array.

        Args:
            values (numpy.ndarray): Values to be quantized.
            value_is_float (bool): Whether the passed values are real (float) values or not.
                If False, the values will be quantized according to the passed scale and zero_point.
            options (QuantizationOptions): Quantization options set
            stats (Optional[MinMaxQuantizationStats]): Quantization batch statistics set
            params (Optional[UniformQuantizationParameters]): Quantization parameters set
                (scale, zero-point)
        """
        if value_is_float:
            if isinstance(values, numpy.ndarray):
                assert_true(
                    numpy.issubdtype(values.dtype, numpy.floating),
                    "Values must be float if value_is_float is set to True, "
                    f"got {values.dtype}: {values}",
                )

            self.values = deepcopy(values) if isinstance(values, numpy.ndarray) else values

            # If no stats are provided, compute them.
            # Note that this cannot be done during tracing
            if stats is None:
                self.quantizer.compute_quantization_stats(values)

            # If the quantization params are not provided, compute them
            # Note that during tracing, this does not use the tracer in any way and just
            # computes some constants.
            if params is None:
                # for mypy
                assert options is not None
                self.quantizer.compute_quantization_parameters(options, self.quantizer.quant_stats)

            # Once the quantizer is ready, quantize the float values provided
            self.quant()
        else:
            assert_true(
                params is not None,
                f"When initializing {self.__class__.__name__} with value_is_float == "
                "False, the scale and zero_point parameters are required.",
            )
            if isinstance(values, numpy.ndarray):
                assert_true(
                    numpy.issubdtype(values.dtype, numpy.integer)
                    or numpy.issubdtype(values.dtype, numpy.unsignedinteger),
                    f"Can't create a QuantizedArray from {values.dtype} values "
                    "when int/uint was required",
                )

            self.qvalues = deepcopy(values) if isinstance(values, numpy.ndarray) else values

            # Populate self.values
            self.dequant()

    def __call__(self) -> Optional[numpy.ndarray]:
        return self.qvalues

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        metadata["quantizer"] = self.quantizer
        metadata["n_bits"] = self.n_bits
        metadata["values"] = self.values
        metadata["qvalues"] = self.qvalues
        return metadata

    @staticmethod
    def load_dict(metadata: Dict) -> QuantizedArray:
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            QuantizedArray: The loaded object.
        """
        obj = QuantizedArray(n_bits=metadata["n_bits"], values=metadata["values"])

        obj.quantizer = metadata["quantizer"]
        obj.values = metadata["values"]
        obj.qvalues = metadata["qvalues"]

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

    def update_values(self, values: numpy.ndarray) -> numpy.ndarray:
        """Update values to get their corresponding qvalues using the related quantized parameters.

        Args:
            values (numpy.ndarray): Values to replace self.values

        Returns:
            qvalues (numpy.ndarray): Corresponding qvalues
        """
        self.values = deepcopy(values) if isinstance(values, numpy.ndarray) else values
        self.quant()
        return self.qvalues

    def update_quantized_values(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Update qvalues to get their corresponding values using the related quantized parameters.

        Args:
            qvalues (numpy.ndarray): Values to replace self.qvalues

        Returns:
            values (numpy.ndarray): Corresponding values
        """
        self.qvalues = deepcopy(qvalues) if isinstance(qvalues, numpy.ndarray) else qvalues
        self.dequant()
        return self.values

    def quant(self) -> Optional[numpy.ndarray]:
        """Quantize self.values.

        Returns:
            numpy.ndarray: Quantized values.
        """

        self.qvalues = self.quantizer.quant(self.values)
        return self.qvalues

    def dequant(self) -> numpy.ndarray:
        """De-quantize self.qvalues.

        Returns:
            numpy.ndarray: De-quantized values.
        """
        self.values = self.quantizer.dequant(self.qvalues)
        assert_true(
            not isinstance(self.values, numpy.ndarray) or self.values.dtype == numpy.float64,
            "De-quantized values must be float64",
        )
        return self.values
