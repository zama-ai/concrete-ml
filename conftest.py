"""PyTest configuration file"""
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union
from concrete.ml.quantization.quantized_module import QuantizedModule

import numpy
import pytest
import torch
from concrete.numpy.compilation import (
    Circuit,
    Configuration,
    configuration,
)
from concrete.numpy.mlir.utils import MAXIMUM_TLU_BIT_WIDTH
from sklearn.datasets import make_classification, make_regression

from concrete.numpy import Graph as CNPGraph


def pytest_addoption(parser):
    """Options for pytest"""

    parser.addoption(
        "--global-coverage-infos-json",
        action="store",
        default=None,
        type=str,
        help="To dump pytest-cov term report to a text file.",
    )

    parser.addoption(
        "--keyring-dir",
        action="store",
        default=None,
        type=str,
        help="Specify the dir to use to store key cache",
    )

    parser.addoption(
        "--forcing_random_seed",
        action="store",
        default=None,
        type=int,
        help="To force the seed of each and every unit test, to be able to "
        "reproduce a particular issue.",
    )

    parser.addoption(
        "--weekly",
        action="store_true",
        help="To do longer tests.",
    )

    parser.addoption(
        "--only_vl_tests",
        action="store_true",
        help="To run only VL tests (i.e., no FHE compilation nor execution).",
    )


DEFAULT_KEYRING_PATH = Path.home().resolve() / ".cache/concrete-ml_pytest"


def get_keyring_dir_from_session_or_default(
    session: Optional[pytest.Session] = None,
) -> Optional[Path]:
    """Get keyring dir from test session."""
    if session is None:
        return DEFAULT_KEYRING_PATH

    keyring_dir = session.config.getoption("--keyring-dir", default=None)
    if keyring_dir is not None:
        if keyring_dir.lower() == "disable":
            return None
        keyring_dir = Path(keyring_dir).expanduser().resolve()
    else:
        keyring_dir = DEFAULT_KEYRING_PATH
    return keyring_dir


@pytest.fixture
def default_keyring_path():
    """Fixture to get test keyring dir."""
    return DEFAULT_KEYRING_PATH


# This is only for doctests where we currently cannot make use of fixtures
original_compilation_config_init = Configuration.__init__


def monkeypatched_compilation_configuration_init_for_codeblocks(
    self: Configuration, *args, **kwargs
):
    """Monkeypatched compilation configuration init for codeblocks tests."""
    original_compilation_config_init(self, *args, **kwargs)
    self.dump_artifacts_on_unexpected_failures = False
    self.enable_unsafe_features = True  # This is for our tests only, never use that in prod
    self.treat_warnings_as_errors = True
    self.use_insecure_key_cache = True  # This is for our tests only, never use that in prod
    self.insecure_key_cache_location = "ConcreteNumpyKeyCache"


def pytest_sessionstart(session: pytest.Session):
    """Handle keyring for session and codeblocks Configuration if needed."""
    if session.config.getoption("--codeblocks", default=False):
        # setattr to avoid mypy complaining
        # Disable the flake8 bug bear warning for the mypy fix
        setattr(  # noqa: B010
            Configuration,
            "__init__",
            monkeypatched_compilation_configuration_init_for_codeblocks,
        )

    keyring_dir = get_keyring_dir_from_session_or_default(session)
    if keyring_dir is None:
        return
    keyring_dir.mkdir(parents=True, exist_ok=True)
    keyring_dir_as_str = str(keyring_dir)
    print(f"Using {keyring_dir_as_str} as key cache dir")
    configuration.insecure_key_cache_location = keyring_dir_as_str


def pytest_sessionfinish(session: pytest.Session, exitstatus):  # pylint: disable=unused-argument
    """Pytest callback when testing ends."""
    # Hacked together from the source code, they don't have an option to export to file and it's too
    # much work to get a PR in for such a little thing
    # https://github.com/pytest-dev/pytest-cov/blob/
    # ec344d8adf2d78238d8f07cb20ed2463d7536970/src/pytest_cov/plugin.py#L329
    if session.config.pluginmanager.hasplugin("_cov"):
        global_coverage_file = session.config.getoption(
            "--global-coverage-infos-json", default=None
        )
        if global_coverage_file is not None:
            cov_plugin = session.config.pluginmanager.getplugin("_cov")
            coverage_txt = cov_plugin.cov_report.getvalue()
            coverage_status = 0
            if (
                cov_plugin.options.cov_fail_under is not None
                and cov_plugin.options.cov_fail_under > 0
            ):
                failed = cov_plugin.cov_total < cov_plugin.options.cov_fail_under
                # If failed is False coverage_status is 0, if True it's 1
                coverage_status = int(failed)
            global_coverage_file_path = Path(global_coverage_file).resolve()
            with open(global_coverage_file_path, "w", encoding="utf-8") as f:
                json.dump({"exit_code": coverage_status, "content": coverage_txt}, f)

    keyring_dir = get_keyring_dir_from_session_or_default(session)
    if keyring_dir is not None:
        # Remove incomplete keys
        for incomplete_keys in keyring_dir.glob("**/*incomplete*"):
            shutil.rmtree(incomplete_keys, ignore_errors=True)


@pytest.fixture
def default_configuration():
    """Return the default test compilation configuration"""

    return Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
        use_insecure_key_cache=True,  # This is for our tests only, never use that in prod
        insecure_key_cache_location="ConcreteNumpyKeyCache",
        jit=True,
        p_error=None,  # To avoid any confusion: we are always using kwarg p_error
        global_p_error=None,  # To avoid any confusion: we are always using kwarg global_p_error
    )


@pytest.fixture
def default_configuration_no_jit():
    """Return the default test compilation configuration"""

    return Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
        use_insecure_key_cache=True,  # This is for our tests only, never use that in prod
        insecure_key_cache_location="ConcreteNumpyKeyCache",
        jit=False,
        p_error=None,  # To avoid any confusion: we are always using kwarg p_error
        global_p_error=None,  # To avoid any confusion: we are always using kwarg global_p_error
    )


REMOVE_COLOR_CODES_RE = re.compile(r"\x1b[^m]*m")


@pytest.fixture
def remove_color_codes():
    """Return the re object to remove color codes"""
    return lambda x: REMOVE_COLOR_CODES_RE.sub("", x)


def function_to_seed_torch(seed):
    """Function to seed torch"""

    # Seed torch with something which is seed by pytest-randomly
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def autoseeding_of_everything(record_property, request):
    """Function to seed everything we can"""
    main_seed = request.config.getoption("--forcing_random_seed", default=None)

    if main_seed is None:
        main_seed = random.randint(0, 2**64 - 1)

    seed = main_seed
    record_property("main seed", main_seed)

    # Python
    random.seed(seed)
    print("\nForcing seed to random.seed to ", seed)
    print(
        f"\nRelaunch the tests with --forcing_random_seed {seed} "
        + "--randomly-dont-reset-seed to reproduce. Remark that adding --randomly-seed=... "
        + "is needed when the testcase uses randoms in pytest parameters"
    )
    print(
        "Remark that potentially, any option used in the pytest call may have an impact so in "
        + "case of problem to reproduce, you may want to have a look to `make pytest` options"
    )

    # Numpy
    seed += 1
    numpy.random.seed(seed % 2**32)

    # Seed torch
    seed += 1
    function_to_seed_torch(seed)
    return {"main seed": main_seed}


@pytest.fixture
def is_weekly_option(request):
    """Function to see if we are in --weekly configuration"""
    is_weekly = request.config.getoption("--weekly")
    return is_weekly


@pytest.fixture
def is_vl_only_option(request):
    """Function to see if we are in --only_vl_tests configuration"""
    only_vl_tests = request.config.getoption("--only_vl_tests")
    return only_vl_tests


# Method is not ideal as some MLIR can contain TLUs but not the associated graph
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2381
def check_graph_input_has_no_tlu_impl(graph: CNPGraph):
    succ = list(graph.graph.successors(graph.input_nodes[0]))
    if any(s.converted_to_table_lookup for s in succ):
        raise AssertionError(f"Graph contains a TLU on an input node: {str(graph.format())}")


# Method is not ideal as some MLIR can contain TLUs but not the associated graph
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2381
def check_graph_output_has_no_tlu_impl(graph: CNPGraph):
    if graph.output_nodes[0].converted_to_table_lookup:
        raise AssertionError(f"Graph output is produced by a TLU: {str(graph.format())}")


# Method is not ideal as some MLIR can contain TLUs but not the associated graph
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2381
def check_graph_has_no_input_output_tlu_impl(graph: CNPGraph):
    check_graph_input_has_no_tlu_impl(graph)
    check_graph_output_has_no_tlu_impl(graph)


# FIXME: To update when the https://github.com/zama-ai/concrete-numpy-internal/issues/1714 feature
# becomes available in CN
def check_circuit_has_no_tlu_impl(circuit: Circuit):
    if "apply_" in circuit.mlir and "_lookup_table" in circuit.mlir:
        raise AssertionError("The circuit contains at least one TLU")


def check_circuit_precision_impl(circuit: Circuit):
    circuit_precision = circuit.graph.maximum_integer_bit_width()
    if circuit_precision > MAXIMUM_TLU_BIT_WIDTH:
        raise AssertionError(
            f"The circuit's precision is expected to be less than {MAXIMUM_TLU_BIT_WIDTH}. "
            f"Got {circuit_precision}."
        )


@pytest.fixture
def check_graph_input_has_no_tlu():
    return check_graph_input_has_no_tlu_impl


@pytest.fixture
def check_graph_output_has_no_tlu():
    return check_graph_output_has_no_tlu_impl


@pytest.fixture
def check_graph_has_no_input_output_tlu():
    return check_graph_has_no_input_output_tlu_impl


@pytest.fixture
def check_circuit_has_no_tlu():
    return check_circuit_has_no_tlu_impl


@pytest.fixture
def check_circuit_precision():
    return check_circuit_precision_impl


def check_array_equality_impl(actual: Any, expected: Any, verbose: bool = True):
    """Assert that `actual` is equal to `expected`."""

    assert numpy.array_equal(actual, expected), (
        ""
        if not verbose
        else f"""

Expected Output
===============
{expected}

Actual Output
=============
{actual}

        """
    )


@pytest.fixture
def check_array_equality():
    """Fixture to check array equality"""

    return check_array_equality_impl


@pytest.fixture
def check_float_arrays_equal():
    """Fixture to check if two float arrays are equal with epsilon precision tolerance"""

    def check_float_arrays_equal_impl(a, b):
        assert numpy.all(numpy.isclose(a, b, rtol=0, atol=0.001))

    return check_float_arrays_equal_impl


@pytest.fixture
def check_r2_score():
    """Fixture to check r2 score"""

    def check_r2_score_impl(expected, actual, acceptance_score=0.99):
        expected = expected.ravel()
        actual = actual.ravel()
        mean_expected = numpy.mean(expected)
        deltas_expected = expected - mean_expected
        deltas_actual = actual - expected
        r2_den = numpy.sum(deltas_expected**2)
        r2_num = numpy.sum(deltas_actual**2)

        # If the values are really close, we consider the test passes
        is_close = numpy.allclose(expected, actual, atol=1e-4, rtol=0)
        if is_close:
            return

        # If the variance of the target values is very low, fix the max allowed for residuals
        # to a known value
        r2_den = max(r2_den, 1e-5)

        r_square = 1 - r2_num / r2_den
        assert (
            r_square >= acceptance_score
        ), f"r2 score of {numpy.round(r_square, 4)} is not high enough."

    return check_r2_score_impl


@pytest.fixture
def check_accuracy():
    """Fixture to check the accuracy"""

    def check_accuracy_impl(expected, actual, threshold=0.9):
        accuracy = numpy.mean(expected == actual)
        assert accuracy >= threshold, f"Accuracy of {accuracy} is not high enough ({threshold})."

    return check_accuracy_impl


@pytest.fixture
def load_data():
    """Fixture for generating random regression or classification problem."""

    def custom_load_data(
        dataset: Union[str, Callable],
        *args,
        strictly_positive: bool = False,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """Generate a random regression or classification problem.

        Sklearn's make_regression() method generates a random regression problem without any domain
        restrictions. However, some models can only handle non negative or (strictly) positive target
        values. This function therefore adapts it in order to make it work for any tested regressors.

        For classifier, Sklearn's make_classification() method is directly called.

        Args:
            dataset (str, Callable): Either "classification" or "regression" generating synthetic
                datasets or a callable for any other dataset generation.
            strictly_positive (bool): If True, the regression data will be only composed of strictly
                positive values. It has no effect on classification problems. Default to False.
            model_name (Optional[str]): If NeuralNetRegressor, a specific change on y will be
                applied.
        """

        # Create a random_state value in order to seed the data generation functions. This enables
        # all tests that use this fixture to be deterministic and thus reproducible.
        random_state = numpy.random.randint(0, 2**15)

        # If the dataset should be generated for a classification problem.
        if dataset == "classification":
            return make_classification(*args, **kwargs, random_state=random_state)

        # If the dataset should be generated for a regression problem.
        elif dataset == "regression":
            generated_regression = list(make_regression(*args, **kwargs, random_state=random_state))

            # Some regressors can only handle positive target values, often strictly positive.
            if strictly_positive:
                generated_regression[1] = numpy.abs(generated_regression[1]) + 1

            if model_name == "NeuralNetRegressor":
                generated_regression[1] = (
                    generated_regression[1].reshape(-1, 1).astype(numpy.float32)
                )

            return tuple(generated_regression)

        # Any other dataset to generate.
        else:
            return dataset()

    return custom_load_data


@pytest.fixture
def check_is_good_execution_for_cml_vs_circuit():
    """Compare quantized module or built-in inference vs Concrete-Numpy circuit."""

    def batch_circuit_inference(inputs: numpy.ndarray, circuit: Circuit):
        """Execute a circuit on a batch of data."""
        # FIXME for now, only allow VL with p_error = 0 since want to make sure the VL
        # (without randomness) matches perfectly CML's predictions.
        # (see https://github.com/zama-ai/concrete-ml-internal/issues/2519)
        if circuit.configuration.virtual and circuit.configuration.p_error not in (
            None,
            0.0,
        ):
            raise ValueError(
                "Virtual Library (VL) can not be tested with a simulated p_error. "
                "Please make sure to have a p_error = 0 or None when the VL is enabled."
            )
        results_cnp_circuit = []
        for i in range(inputs[0].shape[0]):
            q_x = tuple(inputs[input][[i]] for input in range(len(inputs)))
            q_result = circuit.encrypt_run_decrypt(*q_x)
            results_cnp_circuit.append(q_result)
        results_cnp_circuit = numpy.concatenate(results_cnp_circuit, axis=0)
        return results_cnp_circuit

    def check_is_good_execution_for_cml_vs_circuit_impl(
        inputs: Union[tuple, numpy.ndarray],
        model_function: Union[Callable, QuantizedModule],
        n_allowed_runs: int = 5,
    ):
        """Check that a model or a quantized module give the same output as the circuit.

        Args:
            inputs (tuple, numpy.ndarray): inputs for the model.
            model_function (Callable, QuantizedModule): either the Concrete-ML sklearn built-in model
                or a quantized module.
            n_allowed_runs (int): in case of FHE execution randomness can make the output slightly different
                this allows to run the evaluation multiple times
        """
        inputs = (inputs,) if not isinstance(inputs, tuple) else inputs

        for _ in range(n_allowed_runs):

            # Check if model_function is QuantizedModule
            if isinstance(model_function, QuantizedModule):
                # In the case of a quantized module, integer inputs are expected.
                assert numpy.all([numpy.issubdtype(input.dtype, numpy.integer) for input in inputs])
                results_cnp_circuit = batch_circuit_inference(inputs, model_function.fhe_circuit)
                results_model_function = model_function.forward(*inputs)

            elif model_function._is_a_public_cml_model:
                # In the case of a model, floating point inputs are expected.
                assert numpy.all(
                    [numpy.issubdtype(input.dtype, numpy.floating) for input in inputs]
                )
                results_cnp_circuit = model_function.predict(*inputs, execute_in_fhe=True)
                results_model_function = model_function.predict(*inputs, execute_in_fhe=False)
            else:
                raise ValueError(
                    "numpy_function should be a built-in concrete sklearn model or a QuantizedModule object."
                )

            if numpy.array_equal(results_cnp_circuit, results_model_function):
                return
        raise RuntimeError(
            f"Mismatch between circuit results:\n{results_cnp_circuit}\n"
            f"and model function results:\n{results_model_function}"
        )

    return check_is_good_execution_for_cml_vs_circuit_impl
