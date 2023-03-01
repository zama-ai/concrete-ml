"""Tests for ReduceSum operator on a Torch model using leveled and PBS circuits."""

from functools import partial

import numpy
import pytest

from concrete.ml.pytest.torch_models import TorchSum, TorchSumMod
from concrete.ml.torch.compile import compile_torch_model


@pytest.mark.parametrize(
    "data_generator",
    [
        pytest.param(partial(numpy.random.uniform, 0, 1), id="uniform"),
        pytest.param(partial(numpy.random.normal, 0, 1), id="normal"),
        pytest.param(partial(numpy.random.gamma, 1, 2), id="gamma"),
    ],
)
@pytest.mark.parametrize(
    "keepdims", [pytest.param(keepdims, id=f"keepdims-{keepdims}") for keepdims in [True, False]]
)
# For the following tests, we need to make sure all circuits don't reach more than 16 bits of
# precision as some have a PBS.
# Besides, the associated PBS model (TorchSumMod) needs an extra bit when executed, meaning that the
# maximum n_bits value possible to consider is 15, even if a single value is summed
@pytest.mark.parametrize(
    "n_bits, size, axes",
    [
        pytest.param(n_bits, size, axes, id=f"n_bits-{n_bits}-size-{size}-axes-{axes}")
        for (n_bits, size, axes) in [
            (15, (1,), (0,)),
            (10, (50, 1), (0,)),
            (15, (50, 1), (1,)),
            (10, (10, 5), None),
            (10, (10, 10, 50), (2,)),
            (10, (5, 10, 10), (0, 2)),
        ]
    ],
)
@pytest.mark.parametrize(
    "model_class, use_virtual_lib, has_tlu",
    [
        pytest.param(TorchSum, False, False, id="sum_leveled_in_FHE"),
        pytest.param(TorchSumMod, True, True, id="sum_with_pbs_in_VL"),
    ],
)
# pylint: disable-next=too-many-arguments
def test_sum(
    model_class,
    n_bits,
    size,
    axes,
    keepdims,
    use_virtual_lib,
    has_tlu,
    data_generator,
    default_configuration,
    is_vl_only_option,
    check_circuit_has_no_tlu,
    check_circuit_precision,
    check_r2_score,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Tests ReduceSum ONNX operator on a torch model."""

    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Create a Torch module that sums the elements of an array
    # We also need to transform the dim parameter as Torch doesn't handle its value to be None
    # while keepdim being set to True
    dim = tuple(axis for axis in range(len(size))) if axes is None else axes
    torch_model = model_class(dim=dim, keepdim=keepdims)

    # Generate the inputset with several samples
    inputset = data_generator(size=(200,) + size)

    # Compile the torch model
    quantized_numpy_module = compile_torch_model(
        torch_model,
        inputset,
        configuration=default_configuration,
        use_virtual_lib=use_virtual_lib,
        n_bits=n_bits,
    )

    # If the model is expected to have some TLUs, check that the circuit precision is under the
    # maximum allowed value
    if has_tlu:
        check_circuit_precision(quantized_numpy_module.fhe_circuit)

    # Else, check if that the model actually doesn't have any TLUs
    else:
        check_circuit_has_no_tlu(quantized_numpy_module.fhe_circuit)

    # Take an inputset's subset as inputs
    numpy_input = inputset[:5]

    quantized_numpy_module.check_model_is_compiled()

    # Execute the sum in FHE over several samples
    q_result = []
    for numpy_input_i in numpy_input:
        # Quantize the input
        q_input = quantized_numpy_module.quantize_input((numpy_input_i,))
        check_is_good_execution_for_cml_vs_circuit(q_input, quantized_numpy_module)

        if not isinstance(q_input, tuple):
            q_input = (q_input,)

        # Execute the sum in FHE over the sample
        q_result.append(quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*q_input)[0])

    # Dequantize the output
    computed_sum = quantized_numpy_module.dequantize_output(numpy.array(q_result))

    # Compute the expected sum
    # As the calibration inputset and inputs are ran over several samples, we need to apply the
    # sum on all the given axes except the first one (the sample axis), including when axes is
    # set to None (i.e. sum over all axes). The same transformation is done within the ReduceSum
    # operator.
    axis = (
        tuple(axis + 1 for axis in axes)
        if axes is not None
        else tuple(axis for axis in range(1, len(size) + 1))
    )
    expected_sum = numpy.sum(numpy_input, axis=axis, keepdims=keepdims)

    assert computed_sum.shape == expected_sum.shape

    check_r2_score(expected_sum, computed_sum)
