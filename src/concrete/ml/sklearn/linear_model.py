"""Implement sklearn linear model."""
from __future__ import annotations

import copy
import warnings
from typing import Callable, Optional, Tuple

import numpy
import onnx
import sklearn.linear_model
from concrete.common.compilation import CompilationArtifacts, CompilationConfiguration

from ..common.debugging.custom_assert import assert_true
from ..quantization import PostTrainingAffineQuantization
from ..torch.numpy_module import NumpyModule

# Disable pylint to import hummingbird while ignoring the warnings
# pylint: disable=wrong-import-position,wrong-import-order

# Silence hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402


class SklearnLinearModelMixin:
    """A Mixin class for sklearn linear model with FHE."""

    sklearn_alg: Callable

    def __init__(self, *args, n_bits: int = 2, **kwargs):
        """Initialize the FHE linear model.

        Args:
            n_bits (int): The number of bits over which to quantize the model.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.
        """
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits

    # pylint: disable=invalid-name
    def fit(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs) -> None:
        """Fit the FHE linear model.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.
        """
        # Copy X
        X = copy.deepcopy(X)

        # Train
        # mypy does not see the fit from super(). Need to ignore with mypy warning
        super().fit(X, y, *args, **kwargs)  # type: ignore

        # Convert to onnx
        onnx_model = hb_convert(self, backend="onnx", test_input=X).model

        # Remove Cast nodes
        onnx_model = self.clean_graph(onnx_model)

        # Create NumpyModule from onnx model
        numpy_module = NumpyModule(onnx_model)

        # Apply post-training quantization
        post_training = PostTrainingAffineQuantization(
            n_bits=self.n_bits, numpy_model=numpy_module, is_signed=True
        )

        # Calibrate and create quantize module
        self.quantized_module = post_training.quantize_module(X)

    @staticmethod
    def clean_graph(onnx_model: onnx.ModelProto):
        """Clean the graph of the onnx model.

        This will remove the Cast node in the onnx.graph since they
        have no use in the quantized/FHE model.

        Args:
            onnx_model (onnx.ModelProto): the onnx model

        Returns:
            onnx.ModelProto: the cleaned onnx model
        """
        # Remove the input and output nodes
        for node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "Cast":
                assert_true(len(node.attribute) == 1, "Cast node has more than one attribute")
                node_attribute = node.attribute[0]
                assert_true(
                    (node_attribute.name == "to") & (node_attribute.i == onnx.TensorProto.FLOAT)
                )
                new_node = onnx.helper.make_node(
                    "Identity",
                    inputs=[str(node.input[0])],
                    outputs=node.output,
                )
                # Update current node with new_node
                onnx_model.graph.node[node_index].CopyFrom(new_node)
        return onnx_model

    def fit_benchmark(
        self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs
    ) -> Tuple[SklearnLinearModelMixin, sklearn.linear_model.LinearRegression]:
        """Fit the sklearn linear model and the FHE linear model.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.

        Returns:
            Tuple[SklearnLinearModelMixin, sklearn.linear_model.LinearRegression]:
                The FHE and sklearn LinearRegression.
        """
        # Train the sklearn model without X quantized
        sklearn_model = self.sklearn_alg(*args, **kwargs)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        SklearnLinearModelMixin.fit(self, X, y, *args, **kwargs)
        return self, sklearn_model

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict on user data.

        Predict on user data using either the quantized clear model,
        implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit

        Args:
            X (numpy.ndarray): the input data
            execute_in_fhe (bool): whether to execute the inference in FHE

        Returns:
            numpy.ndarray: the prediction as ordinals
        """
        # Quantize the input
        qX = self.quantized_module.quantize_input(X)

        # mypy
        assert isinstance(qX, numpy.ndarray)

        if execute_in_fhe:

            # Make sure the model is compiled
            assert_true(
                self.quantized_module.is_compiled,
                "The model is not compiled. Please run compile(...) first.",
            )

            # mypy
            assert self.quantized_module.forward_fhe is not None
            # mypy does not see teh self.coef_ from sklearn.linear_model.LinearReression.
            # Need to ignore with mypy warning.
            n_targets = 1 if self.coef_.ndim == 1 else self.coef_.shape[0]  # type: ignore
            y_preds = numpy.zeros(shape=(X.shape[0], n_targets))
            # Execute the compiled FHE circuit
            # Create a numpy array with the expected shape: (n_samples, n_classes)
            for i, qX_i in enumerate(qX):
                fhe_pred = self.quantized_module.forward_fhe.run(
                    qX_i.astype(numpy.uint8).reshape(1, qX_i.shape[0])
                )
                y_preds[i, :] = fhe_pred[0]
            # Convert to numpy array
            y_preds = self.quantized_module.dequantize_output(y_preds)
            return y_preds
        return self.quantized_module.forward_and_dequant(qX)

    def compile(
        self,
        X: numpy.ndarray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
    ) -> None:
        """Compile the FHE linear model.

        Args:
            X (numpy.ndarray): The input data.
            compilation_configuration (Optional[CompilationConfiguration]): Configuration object
                to use during compilation
            compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill during
                compilation
            show_mlir (bool): if set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.
        """
        # Quantize the input
        quantized_numpy_inputset = copy.deepcopy(self.quantized_module.q_inputs[0])
        quantized_numpy_inputset.update_values(X)

        # Compile the model
        self.quantized_module.compile(
            quantized_numpy_inputset, compilation_configuration, compilation_artifacts, show_mlir
        )


class LinearRegression(SklearnLinearModelMixin, sklearn.linear_model.LinearRegression):
    """A linear regression model with FHE."""

    def __init__(
        self,
        n_bits=2,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self.n_bits = n_bits
        self.sklearn_alg = sklearn.linear_model.LinearRegression
