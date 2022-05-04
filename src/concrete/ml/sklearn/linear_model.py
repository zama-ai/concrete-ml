"""Implement sklearn linear model."""
from __future__ import annotations

import copy
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy
import onnx
import sklearn.linear_model
from concrete.numpy.compilation.artifacts import CompilationArtifacts
from concrete.numpy.compilation.configuration import CompilationConfiguration

from ..common.debugging.custom_assert import assert_true
from ..onnx.onnx_model_manipulations import (
    keep_following_outputs_discard_others,
    simplify_onnx_model,
)
from ..quantization import PostTrainingAffineQuantization
from ..torch.numpy_module import NumpyModule

# Disable pylint to import hummingbird while ignoring the warnings
# pylint: disable=invalid-name,wrong-import-position,wrong-import-order,too-many-instance-attributes

# Silence hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402


class SklearnLinearModelMixin:
    """A Mixin class for sklearn linear models with FHE."""

    sklearn_alg: Callable

    def __init__(self, *args, n_bits: Union[int, Dict] = 2, **kwargs):
        """Initialize the FHE linear model.

        Args:
            n_bits (int, Dict): Number of bits to quantize the model. If an int is passed for
                n_bits, the value will be used for activation, inputs and weights. If a dict is
                passed, then it should contain "inputs", "weights" and "outputs" keys with
                corresponding number of quantization bits for:
                - inputs : any input data to any layers
                - weights: learned parameters or constants in the network
                - outputs: final model output
                Default to 2.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.
        """
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits

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

        # Initialize the model
        params = self.get_params()  # type: ignore
        params.pop("n_bits", None)
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(X, y, *args, **kwargs)

        # Convert to onnx
        onnx_model = hb_convert(self.sklearn_model, backend="onnx", test_input=X).model

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

    # clean_graph is used by inheritance and the calling self is needed.
    # pylint does not see it and complains that clean_graph should use @staticmethod
    # thus we need to ignore the warning.
    # pylint: disable=no-self-use

    def clean_graph(self, onnx_model: onnx.ModelProto):
        """Clean the graph of the onnx model.

        This will remove the Cast node in the onnx.graph since they
        have no use in the quantized/FHE model.

        Args:
            onnx_model (onnx.ModelProto): the onnx model

        Returns:
            onnx.ModelProto: the cleaned onnx model
        """
        op_type_to_remove = {"Cast", "Softmax", "ArgMax"}

        # Remove the input and output nodes
        for node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type in op_type_to_remove:
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

        simplify_onnx_model(onnx_model)
        return onnx_model

    # pylint: enable=no-self-use

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
            # mypy does not see the self.coef_ from sklearn.linear_model.LinearRegression.
            # Need to ignore with mypy warning.
            n_targets = (
                1 if self.sklearn_model.coef_.ndim == 1 else self.sklearn_model.coef_.shape[0]
            )
            y_preds = numpy.zeros(shape=(X.shape[0], n_targets))
            # Execute the compiled FHE circuit
            # Create a numpy array with the expected shape: (n_samples, n_classes)
            for i, qX_i in enumerate(qX):
                fhe_pred = self.quantized_module.forward_fhe.encrypt_run_decrypt(
                    qX_i.astype(numpy.uint8).reshape(1, qX_i.shape[0])
                )
                y_preds[i, :] = fhe_pred[0]
            # Convert to numpy array
            y_preds = self.quantized_module.dequantize_output(y_preds)
        else:
            y_preds = self.quantized_module.forward_and_dequant(qX)
        return y_preds

    def compile(
        self,
        X: numpy.ndarray,
        configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
    ) -> None:
        """Compile the FHE linear model.

        Args:
            X (numpy.ndarray): The input data.
            configuration (Optional[CompilationConfiguration]): Configuration object
                to use during compilation
            compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill during
                compilation
            show_mlir (bool): if set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.
            use_virtual_lib (bool): whether to compile using the virtual library that allows higher
                bitwidths with simulated FHE computation. Defaults to False
        """
        # Quantize the input
        quantized_numpy_inputset = copy.deepcopy(self.quantized_module.q_inputs[0])
        quantized_numpy_inputset.update_values(X)

        # Compile the model
        self.quantized_module.compile(
            quantized_numpy_inputset,
            configuration,
            compilation_artifacts,
            show_mlir,
            use_virtual_lib=use_virtual_lib,
        )


class LinearRegression(
    SklearnLinearModelMixin, sklearn.base.RegressorMixin, sklearn.base.BaseEstimator
):
    """A linear regression model with FHE."""

    sklearn_alg = sklearn.linear_model.LinearRegression

    def __init__(
        self,
        n_bits=2,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        # FIXME #893
        self.n_bits = n_bits
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        super().__init__(n_bits=n_bits)


class LogisticRegression(SklearnLinearModelMixin, sklearn.linear_model.LogisticRegression):
    """A logistic regression model with FHE."""

    sklearn_alg = sklearn.linear_model.LogisticRegression
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        n_bits=2,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        # FIXME #893
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        super().__init__(n_bits=n_bits)

    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/425:
    # use clean_graph and predict from BaseLinearClassifierMixin
    # but difficulties because we need to make python understand that
    # LogisticRegression.clean_graph must be BaseLinearClassifierMixin.clean_graph
    # and not SklearnLinearModelMixin.clean_graph
    # pylint: disable=duplicate-code
    # pylint: disable=R0801
    def clean_graph(self, onnx_model: onnx.ModelProto):
        nodes_to_remove = []
        output_to_follow = "variable"
        # Find nodes to remove (after the sigmoid)
        sigmoid_reached = False
        for node in onnx_model.graph.node:
            if sigmoid_reached:
                nodes_to_remove.append(node)
            if node.op_type == "Sigmoid":
                sigmoid_reached = True
                # Create output node

                onnx_model.graph.output[0].CopyFrom(
                    onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, [2])
                )
                output_to_follow = node.output[0]

        if sigmoid_reached:
            # Remove nodes
            for node in nodes_to_remove:
                onnx_model.graph.node.remove(node)

        keep_following_outputs_discard_others(onnx_model, [output_to_follow])
        return super().clean_graph(onnx_model)

    def decision_function(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict confidence scores for samples.

        Args:
            X: samples to predict
            execute_in_fhe: if True, the model will be executed in FHE mode

        Returns:
            numpy.ndarray: confidence scores for samples
        """
        y_preds = super().predict(X, execute_in_fhe)
        return y_preds

    def predict_proba(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: samples to predict
            execute_in_fhe: if True, the model will be executed in FHE mode

        Returns:
            numpy.ndarray: class probabilities for samples
        """
        y_preds = self.decision_function(X, execute_in_fhe)
        if y_preds.shape[1] == 1:
            # Sigmoid already applied in the graph
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
        else:
            y_preds = numpy.exp(y_preds)
            y_preds = y_preds / numpy.sum(y_preds, axis=1, keepdims=True)
        return y_preds

    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/425:
    # use clean_graph and predict from BaseLinearClassifierMixin
    # but difficulties because we need to make python understand that
    # LogisticRegression.clean_graph must be BaseLinearClassifierMixin.clean_graph
    # and not SklearnLinearModelMixin.clean_graph
    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/375: we need to refacto
    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        y_preds = self.predict_proba(X, execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds

    # pylint: enable=duplicate-code
    # pylint: enable=R0801
    # pylint: enable=too-many-instance-attributes
    # pylint: enable=invalid-name
