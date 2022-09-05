"""Quantization utilities for a numpy array/tensor."""

from copy import deepcopy
from typing import Optional, get_type_hints

import numpy

from ..common.debugging import assert_true

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
    is_signed: bool = False
    is_symmetric: bool = False
    is_qat: bool = False

    def __init__(
        self, n_bits, is_signed: bool = False, is_symmetric: bool = False, is_qat: bool = False
    ) -> None:
        self.n_bits = n_bits
        self.is_signed = is_signed
        self.is_symmetric = is_symmetric
        self.is_qat = is_qat

        # QAT quantization is not symmetric
        assert_true(not self.is_qat or (self.is_qat and not self.is_symmetric))

        # Symmetric quantization is signed
        assert_true(not self.is_symmetric or (self.is_signed and self.is_symmetric))

    def copy_opts(self, opts):
        """Copy the options from a different structure.

        Args:
            opts (QuantizationOptions): structure to copy parameters from.
        """

        self.n_bits = opts.n_bits
        self.is_signed = opts.is_signed
        self.is_symmetric = opts.is_symmetric
        self.is_qat = opts.is_qat

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


class MinMaxQuantizationStats:
    """Calibration set statistics.

    This class stores the statistics for the calibration set or for a calibration data batch.
    Currently we only store min/max to determine the quantization range. The min/max are computed
    from the calibration set.
    """

    rmax: Optional[float] = None
    rmin: Optional[float] = None
    uvalues: Optional[numpy.ndarray] = None

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


class UniformQuantizationParameters:
    """Quantization parameters for uniform quantization.

    This class stores the parameters used for quantizing real values to discrete integer values.
    The parameters are computed from quantization options and quantization statistics.
    """

    scale: Optional[float] = None
    zero_point: Optional[int] = None
    offset: Optional[int] = None

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
                self.scale = 1.0
                # Ideally we should get rid of round here but it is risky
                # regarding the FHE compilation.
                # Indeed, the zero_point value for the weights has to be an integer
                # for the compilation to work.
                self.zero_point = numpy.rint(-stats.rmin).astype(numpy.int64)
            else:
                # If the value is not a 0 we can tweak the scale factor so that
                # the value quantizes to 1
                self.scale = stats.rmax
                self.zero_point = 0
        else:
            if options.is_symmetric:
                assert_true(not options.is_qat)
                self.zero_point = 0
                self.scale = numpy.maximum(numpy.abs(stats.rmax), numpy.abs(stats.rmin)) / (
                    (2**options.n_bits - 1 - self.offset)
                )
            else:
                if (
                    options.is_qat
                    and stats.uvalues is not None
                    and stats.uvalues.size <= 2**options.n_bits
                ):
                    # FIXME: this crashes when a model is poorly trained
                    # the code crashes later on without this check
                    # https://github.com/zama-ai/concrete-ml-internal/issues/1620
                    assert_true(
                        len(stats.uvalues) > 1,
                        "A single unique value was detected in a tensor of "
                        "quantized values in a QAT import.\n"
                        "Please check the stability thresholds.\n"
                        "This can occur with a badly trained model.",
                    )
                    unique_scales = numpy.unique(numpy.diff(stats.uvalues))
                    self.scale = unique_scales[0]

                if self.scale is None:
                    self.scale = (
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
                            stats.rmax * (-self.offset)
                            - (stats.rmin * (2**options.n_bits - 1 - self.offset))
                        )
                        / (stats.rmax - stats.rmin)
                    ).astype(numpy.int64)


# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1434
# Change UniformQuantizer inheritance from UniformQuantizationParameters to composition.


class UniformQuantizer(UniformQuantizationParameters, QuantizationOptions, MinMaxQuantizationStats):
    """Uniform quantizer.

    Contains all information necessary for uniform quantization and provides
    quantization/dequantization functionality on numpy arrays.

    Args:
        options (QuantizationOptions): Quantization options set
        stats (Optional[MinMaxQuantizationStats]): Quantization batch statistics set
        params (Optional[UniformQuantizationParameters]): Quantization parameters set
            (scale, zero-point)
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        options: QuantizationOptions = None,
        stats: Optional[MinMaxQuantizationStats] = None,
        params: Optional[UniformQuantizationParameters] = None,
        **kwargs,
    ):
        if options is not None:
            self.copy_opts(options)

        if stats is not None:
            self.copy_stats(stats)

        if params is not None:
            self.copy_params(params)

        if kwargs:
            self.options, kwargs = fill_from_kwargs(self, QuantizationOptions, **kwargs)
            self.stats, kwargs = fill_from_kwargs(self, MinMaxQuantizationStats, **kwargs)
            self.params, kwargs = fill_from_kwargs(self, UniformQuantizationParameters, **kwargs)

        # All kwargs should belong to one of the parameter sets, anything else is unsupported
        assert_true(len(kwargs) == 0, f"Unexpected kwargs: {kwargs}")

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

        # If the values are produced by a QAT layer, they should have values in the correct
        # range. But we can't know if the QAT quantization was symmetric or not, so the offset
        # is not set up properly. Thus, we don't clip for QAT values
        if not self.is_qat:
            qvalues = qvalues.clip(-self.offset, 2 ** (self.n_bits) - 1 - self.offset)

        return qvalues.astype(numpy.int64)

    def dequant(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Dequantize values.

        Args:
            qvalues (numpy.ndarray): integer values to dequantize

        Returns:
            numpy.ndarray: Dequantized float values.
        """

        # for mypy
        assert self.zero_point is not None
        assert self.scale is not None

        return self.scale * (qvalues + -(float(self.zero_point)))


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

    STABILITY_CONST = 10**-6

    quantizer: UniformQuantizer
    values: numpy.ndarray
    qvalues: numpy.ndarray

    def __init__(
        self,
        n_bits,
        values: Optional[numpy.ndarray],
        value_is_float: bool = True,
        options: QuantizationOptions = None,
        stats: Optional[MinMaxQuantizationStats] = None,
        params: Optional[UniformQuantizationParameters] = None,
        **kwargs,
    ):
        # If no options were passed, create a default options structure with the required n_bits
        options = deepcopy(options) if options is not None else QuantizationOptions(n_bits)

        # Override the options number of bits if an options structure was provided
        # with the number of bits specified by the caller.
        options.n_bits = n_bits

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

            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/303
            # To be seen what should be done in the long run (refactor of this class or Tracers)
            self.values = deepcopy(values) if isinstance(values, numpy.ndarray) else values

            # If no stats are provided, compute them.
            # Note that this can not be done during tracing
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

            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/303
            # To be seen what should be done in the long run (refactor of this class or Tracers)
            self.qvalues = deepcopy(values) if isinstance(values, numpy.ndarray) else values

            # Populate self.values
            self.dequant()

    def __call__(self) -> Optional[numpy.ndarray]:
        return self.qvalues

    def update_values(self, values: numpy.ndarray) -> numpy.ndarray:
        """Update values to get their corresponding qvalues using the related quantized parameters.

        Args:
            values (numpy.ndarray): Values to replace self.values

        Returns:
            qvalues (numpy.ndarray): Corresponding qvalues
        """
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/303
        # To be seen what should be done in the long run (refactor of this class or Tracers)
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
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/303
        # To be seen what should be done in the long run (refactor of this class or Tracers)
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
        """Dequantize self.qvalues.

        Returns:
            numpy.ndarray: Dequantized values.
        """
        # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/721
        # remove this + (-x) when the above issue is done
        self.values = self.quantizer.dequant(self.qvalues)
        assert_true(
            not isinstance(self.values, numpy.ndarray) or self.values.dtype == numpy.float64,
            "Dequantized values must be float64",
        )
        return self.values
