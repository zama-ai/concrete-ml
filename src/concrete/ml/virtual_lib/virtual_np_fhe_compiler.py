"""Virtual Compiler code."""

# FIXME, Concrete Numpy 0.6 integration, #795
# pylint: disable=E1101

from typing import Any, Callable, Dict, Optional, Union, cast

# FIXME, Concrete Numpy 0.6 integration, #795
# from concrete.common.mlir.utils import check_graph_values_compatibility_with_mlir
from concrete.numpy.compilation import (
    CompilationArtifacts,
    CompilationConfiguration,
    Compiler,
    EncryptionStatus,
)

from ..common.debugging import assert_true
from .virtual_fhe_circuit import VirtualCircuit


class VirtualCompiler(Compiler):
    """Class simulating Compiler behavior in the clear, without any actual FHE computations."""

    def __init__(
        self,
        function_to_compile: Callable,
        function_parameters_encrypted_status: Dict[str, Union[str, EncryptionStatus]],
        configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
    ) -> None:
        assert_true(
            configuration is not None,
            "Using the virtual lib requires a CompilationConfiguration.",
            ValueError,
        )
        # for mypy
        configuration = cast(CompilationConfiguration, configuration)
        assert_true(
            configuration.enable_unsafe_features,
            "Using the virtual lib requires enabling unsafe features " "in configuration.",
            ValueError,
        )

        super().__init__(
            function_to_compile,
            function_parameters_encrypted_status,
            configuration,
            compilation_artifacts,
        )

    def get_compiled_fhe_circuit(self, show_mlir: bool = False) -> VirtualCircuit:
        """Return a compiled VirtualCircuit if the instance was evaluated on an inputset.

        Args:
            show_mlir (bool): ignored in this virtual overload. Defaults to False.

        Returns:
            VirtualCircuit: the compiled VirtualCircuit
        """
        self._eval_on_current_inputset()

        assert_true(
            self._graph is not None,
            "Requested VirtualCircuit but no Graph was compiled. "
            f"Did you forget to evaluate {self.__class__.__name__} over an inputset?",
            RuntimeError,
        )

        # FIXME, Concrete Numpy 0.6 integration, #795
        print("FIXME, remove this, #795", show_mlir)

        # We don't have the compilation check to verify that nodes are Integer -> Integer so we need
        # to check ourselves here
        # FIXME, Concrete Numpy 0.6 integration, #795
        # offending_nodes = check_graph_values_compatibility_with_mlir(self._graph)
        offending_nodes: Dict[Any, Any] = {}
        assert_true(
            offending_nodes is None,
            "function you are trying to compile isn't supported for MLIR lowering\n\n"
            + self.graph.format(highlighted_nodes=offending_nodes),
            RuntimeError,
        )

        return VirtualCircuit(self._graph)
