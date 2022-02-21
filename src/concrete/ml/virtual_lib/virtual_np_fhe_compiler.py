"""Virtual NPFHECompiler code."""

from typing import Callable, Dict, Optional, Union, cast

from concrete.common.compilation import CompilationArtifacts, CompilationConfiguration
from concrete.common.debugging import format_operation_graph
from concrete.common.mlir.utils import check_graph_values_compatibility_with_mlir
from concrete.numpy.np_fhe_compiler import EncryptedStatus, NPFHECompiler

from ..common.debugging import assert_true
from .virtual_fhe_circuit import VirtualFHECircuit


class VirtualNPFHECompiler(NPFHECompiler):
    """Class simulating NPFHECompiler behavior in the clear, without any actual FHE computations."""

    def __init__(
        self,
        function_to_compile: Callable,
        function_parameters_encrypted_status: Dict[str, Union[str, EncryptedStatus]],
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
    ) -> None:
        assert_true(
            compilation_configuration is not None,
            "Using the virtual lib requires a CompilationConfiguration.",
            ValueError,
        )
        # for mypy
        compilation_configuration = cast(CompilationConfiguration, compilation_configuration)
        assert_true(
            compilation_configuration.enable_unsafe_features,
            "Using the virtual lib requires enabling unsafe features "
            "in compilation_configuration.",
            ValueError,
        )

        super().__init__(
            function_to_compile,
            function_parameters_encrypted_status,
            compilation_configuration,
            compilation_artifacts,
        )

    def get_compiled_fhe_circuit(self, show_mlir: bool = False) -> VirtualFHECircuit:
        """Return a compiled VirtualFHECircuit if the instance was evaluated on an inputset.

        Args:
            show_mlir (bool): ignored in this virtual overload. Defaults to False.

        Returns:
            VirtualFHECircuit: the compiled VirtualFHECircuit
        """
        self._eval_on_current_inputset()

        assert_true(
            self._op_graph is not None,
            "Requested VirtualFHECircuit but no OPGraph was compiled. "
            f"Did you forget to evaluate {self.__class__.__name__} over an inputset?",
            RuntimeError,
        )

        # We don't have the compilation check to verify that nodes are Integer -> Integer so we need
        # to check ourselves here
        offending_nodes = check_graph_values_compatibility_with_mlir(self._op_graph)
        assert_true(
            offending_nodes is None,
            "function you are trying to compile isn't supported for MLIR lowering\n\n"
            + format_operation_graph(self._op_graph, highlighted_nodes=offending_nodes),
            RuntimeError,
        )

        return VirtualFHECircuit(self._op_graph)
