"""Implement the sklearn models."""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import concrete.numpy as hnp
import numpy
from concrete.common.compilation.artifacts import CompilationArtifacts
from concrete.common.compilation.configuration import CompilationConfiguration
from concrete.common.fhe_circuit import FHECircuit
from sklearn import tree

from ..common.debugging.custom_assert import assert_true
from ..quantization.quantized_array import QuantizedArray
from ._tree_to_tensors import tree_to_numpy


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name
class DecisionTreeClassifier(tree.DecisionTreeClassifier):
    """Implements the sklearn DecisionTreeClassifier."""

    q_x_byfeatures: list
    fhe_tree: FHECircuit

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        n_bits: int = 6,
    ):
        """Initialize the DecisionTreeClassifier.

        Args:
            criterion: FIXME
            splitter: FIXME
            max_depth: FIXME
            min_samples_split: FIXME
            min_samples_leaf: FIXME
            min_weight_fraction_leaf: FIXME
            max_features: FIXME
            random_state: FIXME
            max_leaf_nodes: FIXME
            min_impurity_decrease: FIXME
            class_weight: FIXME
            ccp_alpha: FIXME
            n_bits: FIXME

        """
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
        self.q_x_byfeatures = []
        self.n_bits = n_bits
        self._tensor_tree_predict = None
        self.fhe_tree = None

    # pylint: enable=too-many-arguments

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs):
        """Fit the sklearn DecisionTreeClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: args for super().fit
            **kwargs: kwargs for super().fit
        """
        # Deepcopy X as we don't want to alterate original values.
        X = copy.deepcopy(X)
        # Check that there are only 2 classes
        assert_true(len(set(y)) == 2, "Only 2 classes are supported currently.")
        self.q_x_byfeatures = []
        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i])
            self.q_x_byfeatures.append(q_x_)
            X[:, i] = q_x_.qvalues

        # Fit the model
        super().fit(X, y, *args, **kwargs)

        # Hummingbird part TODO: replace with a generic method (see issue #126)
        self._tensor_tree_predict = tree_to_numpy(self, X.shape[1], classes=[0, 1])

    def fit_benchmark(
        self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs
    ) -> Tuple[DecisionTreeClassifier, tree.DecisionTreeClassifier]:
        """Fit the sklearn DecisionTreeClassifier and the FHE DecisionTreeClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: args for super().fit
            **kwargs: kwargs for super().fit

        Returns:
            Tuple[DecisionTreeClassifier, tree.DecisionTreeClassifier]:
                                                The FHE and sklearn DecisionTreeClassifier.
        """
        # Train the sklearn model without X quantized
        sklearn_model = super().fit(X, y, *args, **kwargs)
        # Train the FHE model
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model

    # pylint: disable=arguments-differ
    def predict(
        self, X: numpy.ndarray, check_input: bool = True, use_fhe: bool = False
    ) -> numpy.ndarray:
        """Predict using with sklearn.

        Args:
            X (numpy.ndarray): the input data
            check_input(bool): whether to check the input
            use_fhe(bool): FIXME

        Returns:
            the prediction

        """
        # Quantize the input
        X = self.quantize_input(X)

        if use_fhe:
            # Check that self.fhe_tree is not None
            assert_true(
                self.fhe_tree is not None,
                f"You must call {self.compile.__name__} "
                f"before calling {self.predict.__name__} with use_fhe=True.",
            )
            y_preds_ = []
            for i in range(X.shape[0]):
                fhe_pred = self.fhe_tree.run(X[i].astype(numpy.uint8))
                y_preds_.append(fhe_pred)
            # Convert to numpy array
            y_preds = numpy.array(y_preds_)
            y_preds = y_preds[:, 1]
            return y_preds
        return super().predict(X, check_input)

    # pylint: enable=arguments-differ

    def _predict_with_tensors(self, X: numpy.ndarray) -> numpy.ndarray:
        """Predict using the tensors.

        Mainly used for debugging.

        Args:
            X: The input data.

        Returns:
            numpy.ndarray: The prediction.
        """
        # for mypy
        assert self._tensor_tree_predict is not None, "You must fit the model before using it."

        X = self.quantize_input(X)

        # Check that the predictions are correct using tensors
        y_pred = self._tensor_tree_predict(X)[:, 1]
        return y_pred

    def quantize_input(self, X: numpy.ndarray):
        """Quantize the input.

        Args:
            X (numpy.ndarray): the input

        Returns:
            the quantized input
        """
        # Deepcopy to not alter X
        X = copy.deepcopy(X)

        # Quantize using the learned quantization parameters for each feature
        for i, q_x_ in enumerate(self.q_x_byfeatures):
            X[:, i] = q_x_.update_values(X[:, i])
        return X.astype(numpy.uint8)

    def compile(
        self,
        X: numpy.ndarray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
    ):
        """Compile the model.

        Args:
            X (numpy.ndarray): the unquantized dataset
            compilation_configuration (Optional[CompilationConfiguration]): the options for
                compilation
            compilation_artifacts (Optional[CompilationArtifacts]): artifacts object to fill
                during compilation
            show_mlir (bool): whether or not to show MLIR during the compilation

        """
        # Make sure that self.tree_predict is not None
        assert_true(
            self._tensor_tree_predict is not None, "You must fit the model before compiling it."
        )

        X = self.quantize_input(X)
        compiler = hnp.NPFHECompiler(
            self._tensor_tree_predict,
            {"inputs": "encrypted"},
            compilation_configuration,
            compilation_artifacts,
        )
        self.fhe_tree = compiler.compile_on_inputset((sample for sample in X), show_mlir)
