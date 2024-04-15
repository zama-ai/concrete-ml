"""Tests for ReduceSum operator on a Torch model using leveled circuits."""

from functools import partial

import numpy
import pytest

from concrete.ml.pytest.torch_models import TorchSum
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
# In Concrete ML, we consider that all inputs' first dimension should be a batch size
# even in single batch cases. This is why the following test parameters are considering axes that
# are sometimes equal to the input size's dimension, as the batch size is added within the
# test itself.
# Additionally, the axis parameter should neither be None nor contain axis 0 as this dimension is
# used for batching the inference
@pytest.mark.parametrize(
    "n_bits, size, dim",
    [
        pytest.param(n_bits, size, dim, id=f"n_bits-{n_bits}-size-{size}-dim-{dim}")
        for (n_bits, size, dim) in [
            (20, (1,), (1,)),
            (15, (50, 1), (1,)),
            (20, (50, 1), (2,)),
            (15, (10, 10, 50), (3,)),
            (15, (5, 10, 10), (1, 3)),
        ]
    ],
)
def test_sum(
    n_bits,
    size,
    dim,
    keepdims,
    data_generator,
    default_configuration,
    check_circuit_has_no_tlu,
    check_r2_score,
):
    """Tests ReduceSum ONNX operator on a torch model."""

    # Generate the input-set with several samples. This adds a necessary batch size
    inputset = data_generator(size=(100,) + size)

    # Create a Torch module that sums the elements of an array
    torch_model = TorchSum(dim=dim, keepdim=keepdims)

    # Compile the torch model
    quantized_module = compile_torch_model(
        torch_model,
        inputset,
        configuration=default_configuration,
        n_bits=n_bits,
    )

    # Check if that the circuit is fully levelled
    check_circuit_has_no_tlu(quantized_module.fhe_circuit)

    # Take an input-set's subset as inputs
    numpy_input = inputset[:5]

    quantized_module.check_model_is_compiled()

    # Compute the sum, in FHE or with simulation
    computed_sum = quantized_module.forward(numpy_input, fhe="execute")
    assert isinstance(computed_sum, numpy.ndarray)

    # Compute the expected sum
    expected_sum = numpy.sum(numpy_input, axis=dim, keepdims=keepdims)

    assert computed_sum.shape == expected_sum.shape, (
        f"Mismatch found in output shapes. Got {computed_sum.shape} but expected "
        f"{expected_sum.shape}."
    )

    check_r2_score(expected_sum, computed_sum)
