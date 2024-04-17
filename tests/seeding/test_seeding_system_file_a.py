"""Tests for the torch to numpy module."""

import random

import numpy
import pytest


def body():
    """Common function used in the tests, which picks random from numpy in different ways."""
    numpy.set_printoptions(threshold=10000)
    numpy.set_printoptions(linewidth=10000)

    print("Output: ", end="")

    # Python random
    for _ in range(1):
        print(random.randint(0, 1000), end="")

    # Numpy random
    for _ in range(1):
        print(numpy.random.randint(0, 1000), end="")
        print(numpy.random.uniform(-100, 100, size=(3, 3)).flatten(), end="")


@pytest.mark.parametrize("parameters", [1, 2, 3])
def test_bcm_seed_1(parameters):
    """Test python and numpy seeding."""

    assert parameters is not None
    body()


@pytest.mark.parametrize("parameters", [1, 3])
def test_bcm_seed_2(parameters):
    """Test python and numpy seeding."""

    assert parameters is not None
    body()


@pytest.mark.parametrize("parameters", ["a", "b"])
def test_bcm_seed_3(parameters):
    """Test python and numpy seeding."""

    assert parameters is not None
    body()


def test_bcm_seed_4():
    """Test python and numpy seeding."""

    body()
