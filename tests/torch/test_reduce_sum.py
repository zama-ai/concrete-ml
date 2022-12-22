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
@pytest.mark.parametrize(
    "size, axes",
    [
        pytest.param(size, axes, id=f"size-{size}-axes-{axes}")
        for (size, axes) in [
            ((1,), (0,)),
            ((50, 1), (0,)),
            ((50, 1), (1,)),
            ((10, 5), None),
            ((10, 10, 50), (2,)),
            ((5, 10, 10), (0, 2)),
        ]
    ],
)
@pytest.mark.parametrize(
    "model, n_bits, use_virtual_lib, has_tlu",
    [
        pytest.param(TorchSum, 10, False, False, id="sum_leveled_in_FHE"),
        pytest.param(TorchSumMod, 10, True, True, id="sum_with_pbs_in_VL"),
    ],
)
# pylint: disable-next=too-many-arguments
def test_sum(
    model,
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
    torch_model = model(dim=dim, keepdim=keepdims)

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

    assert quantized_numpy_module.is_compiled, "Torch model is not compiled"

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
