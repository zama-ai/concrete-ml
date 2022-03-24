"""Implement the sklearn tree models."""
from __future__ import annotations

from typing import Callable, Optional

import concrete.numpy as hnp
import numpy
import sklearn
from concrete.common.compilation.artifacts import CompilationArtifacts
from concrete.common.compilation.configuration import CompilationConfiguration
from concrete.common.fhe_circuit import FHECircuit

from ..common.debugging.custom_assert import assert_true
from ..common.utils import generate_proxy_function
from ..quantization.quantized_array import QuantizedArray
from ..virtual_lib import VirtualNPFHECompiler
from .base import BaseTreeEstimatorMixin
from .tree_to_numpy import tree_to_numpy

# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name


class DecisionTreeClassifier(sklearn.tree.DecisionTreeClassifier, BaseTreeEstimatorMixin):
    """Implements the sklearn DecisionTreeClassifier."""

    sklearn_alg = sklearn.tree.DecisionTreeClassifier
    q_x_byfeatures: list
    fhe_tree: FHECircuit
    _tensor_tree_predict: Optional[Callable]
    q_y: QuantizedArray

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
        ccp_alpha: float = 0.0,
        n_bits: int = 6,
    ):
        """Initialize the DecisionTreeClassifier.

        # noqa: DAR101

        Args:
            criterion : {"gini", "entropy"}, default="gini"
                The function to measure the quality of a split. Supported criteria are
                "gini" for the Gini impurity and "entropy" for the information gain.

            splitter : {"best", "random"}, default="best"
                The strategy used to choose the split at each node. Supported
                strategies are "best" to choose the best split and "random" to choose
                the best random split.

            max_depth : int, default=None
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure or until all leaves contain less than
                min_samples_split samples.

            min_samples_split : int or float, default=2
                The minimum number of samples required to split an internal node:

                - If int, then consider `min_samples_split` as the minimum number.
                - If float, then `min_samples_split` is a fraction and
                `ceil(min_samples_split * n_samples)` are the minimum
                number of samples for each split.

                .. versionchanged:: 0.18
                Added float values for fractions.

            min_samples_leaf : int or float, default=1
                The minimum number of samples required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at
                least ``min_samples_leaf`` training samples in each of the left and
                right branches.  This may have the effect of smoothing the model,
                especially in regression.

                - If int, then consider `min_samples_leaf` as the minimum number.
                - If float, then `min_samples_leaf` is a fraction and
                `ceil(min_samples_leaf * n_samples)` are the minimum
                number of samples for each node.

                .. versionchanged:: 0.18
                Added float values for fractions.

            min_weight_fraction_leaf : float, default=0.0
                The minimum weighted fraction of the sum total of weights (of all
                the input samples) required to be at a leaf node. Samples have
                equal weight when sample_weight is not provided.

            max_features : int, float or {"auto", "sqrt", "log2"}, default=None
                The number of features to consider when looking for the best split:

                    - If int, then consider `max_features` features at each split.
                    - If float, then `max_features` is a fraction and
                    `int(max_features * n_features)` features are considered at each
                    split.
                    - If "auto", then `max_features=sqrt(n_features)`.
                    - If "sqrt", then `max_features=sqrt(n_features)`.
                    - If "log2", then `max_features=log2(n_features)`.
                    - If None, then `max_features=n_features`.

                Note: the search for a split does not stop until at least one
                valid partition of the node samples is found, even if it requires to
                effectively inspect more than ``max_features`` features.

            random_state : int, RandomState instance or None, default=None
                Controls the randomness of the estimator. The features are always
                randomly permuted at each split, even if ``splitter`` is set to
                ``"best"``. When ``max_features < n_features``, the algorithm will
                select ``max_features`` at random at each split before finding the best
                split among them. But the best found split may vary across different
                runs, even if ``max_features=n_features``. That is the case, if the
                improvement of the criterion is identical for several splits and one
                split has to be selected at random. To obtain a deterministic behaviour
                during fitting, ``random_state`` has to be fixed to an integer.
                See :term:`Glossary <random_state>` for details.

            max_leaf_nodes : int, default=None
                Grow a tree with ``max_leaf_nodes`` in best-first fashion.
                Best nodes are defined as relative reduction in impurity.
                If None then unlimited number of leaf nodes.

            min_impurity_decrease : float, default=0.0
                A node will be split if this split induces a decrease of the impurity
                greater than or equal to this value.

                The weighted impurity decrease equation is the following::

                    N_t / N * (impurity - N_t_R / N_t * right_impurity
                                        - N_t_L / N_t * left_impurity)

                where ``N`` is the total number of samples, ``N_t`` is the number of
                samples at the current node, ``N_t_L`` is the number of samples in the
                left child, and ``N_t_R`` is the number of samples in the right child.

                ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
                if ``sample_weight`` is passed.

                .. versionadded:: 0.19

            class_weight : dict, list of dict or "balanced", default=None
                Weights associated with classes in the form ``{class_label: weight}``.
                If None, all classes are supposed to have weight one. For
                multi-output problems, a list of dicts can be provided in the same
                order as the columns of y.

                Note that for multioutput (including multilabel) weights should be
                defined for each class of every column in its own dict. For example,
                for four-class multilabel classification weights should be
                [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
                [{1:1}, {2:5}, {3:1}, {4:1}].

                The "balanced" mode uses the values of y to automatically adjust
                weights inversely proportional to class frequencies in the input data
                as ``n_samples / (n_classes * numpy.bincount(y))``

                For multi-output, the weights of each column of y will be multiplied.

                Note that these weights will be multiplied with sample_weight (passed
                through the fit method) if sample_weight is specified.

            ccp_alpha : non-negative float, default=0.0
                Complexity parameter used for Minimal Cost-Complexity Pruning. The
                subtree with the largest cost complexity that is smaller than
                ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
                :ref:`minimal_cost_complexity_pruning` for details.

                .. versionadded:: 0.22

            n_bits : int, default=6
                Number of bits used to quantize the input data.
        """
        sklearn.tree.DecisionTreeClassifier.__init__(
            self,
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
        BaseTreeEstimatorMixin.__init__(self, n_bits=n_bits)
        self.q_x_byfeatures = []
        self.n_bits = n_bits
        self.fhe_tree = None

        self.init_args = {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "class_weight": class_weight,
            "random_state": random_state,
            "min_impurity_decrease": min_impurity_decrease,
            "ccp_alpha": ccp_alpha,
        }

    # pylint: enable=too-many-arguments

    def fit(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs):
        """Fit the sklearn DecisionTreeClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: args for super().fit
            **kwargs: kwargs for super().fit
        """
        qX = numpy.zeros_like(X)
        # Check that there are only 2 classes
        assert_true(
            len(numpy.unique(numpy.asarray(y).flatten())) == 2,
            "Only 2 classes are supported currently.",
        )
        # Check that the classes are 0 and 1
        assert_true(
            bool(numpy.all(numpy.unique(y.ravel()) == [0, 1])),
            "y must be in [0, 1]",
        )
        self.q_x_byfeatures = []
        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i])
            self.q_x_byfeatures.append(q_x_)
            qX[:, i] = q_x_.qvalues.astype(numpy.int32)

        # Fit the model
        super().fit(qX, y, *args, **kwargs)

        # Tree inference to numpy
        self._tensor_tree_predict, self.q_y = tree_to_numpy(
            self, qX, "sklearn", output_n_bits=self.n_bits, use_workaround_for_transpose=True
        )

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        # mypy
        assert self.q_y is not None
        y_preds = self.q_y.update_quantized_values(y_preds)
        y_preds = numpy.squeeze(y_preds)
        assert_true(y_preds.ndim > 1, "y_preds should be a 2D array")
        # Check if values are already probabilities
        if any(numpy.abs(numpy.sum(y_preds, axis=1) - 1) > 1e-4):
            # Apply softmax
            # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/518, remove no-cover's
            y_preds = numpy.exp(y_preds)  # pragma: no cover
            y_preds = y_preds / y_preds.sum(axis=1, keepdims=True)  # pragma: no cover
        return y_preds

    # pylint: disable=arguments-differ
    # DecisionTreeClassifier needs a check_input arg which differs from the superclass.
    # Disabling mypy warning for this.
    def predict_proba(  # type: ignore
        self,
        X: numpy.ndarray,
        check_input: Optional[bool] = True,
        execute_in_fhe: Optional[bool] = False,
    ) -> numpy.ndarray:
        """Predict class probabilities of the input samples X.

        Args:
            X (numpy.ndarray): The input data.
            check_input (Optional[bool]): check if the input conforms to shape and
                dtype constraints.
            execute_in_fhe (bool, optional): If True, the predictions are computed in FHE.
                If False, the predictions are computed in the sklearn model. Defaults to False.

        Returns:
            numpy.ndarray: The class probabilities of the input samples X.
        """
        X = self._validate_X_predict(X, check_input)
        if execute_in_fhe:
            y_preds = self._execute_in_fhe(X)
        else:
            y_preds = self._predict_with_tensors(X)
        y_preds = self.post_processing(y_preds)
        return y_preds

    # DecisionTreeClassifier needs a check_input arg which differs from the superclass.
    # Disabling mypy warning for this.
    def predict(  # type: ignore
        self, X: numpy.ndarray, check_input: bool = True, execute_in_fhe: bool = False
    ) -> numpy.ndarray:
        """Predict on user data.

        Predict on user data using either the quantized clear model,
        implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit

        Args:
            X (numpy.ndarray): the input data
            check_input (bool): check if the input conforms to shape and dtype constraints
            execute_in_fhe (bool): whether to execute the inference in FHE

        Returns:
            the prediction as ordinals
        """
        X = self._validate_X_predict(X, check_input)
        y_preds = self.predict_proba(X, check_input, execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds

    # pylint: enable=arguments-differ

    def _execute_in_fhe(self, X: numpy.ndarray) -> numpy.ndarray:
        """Execute the FHE inference on the input data.

        Args:
            X (numpy.ndarray): the input data

        Returns:
            numpy.ndarray: the prediction as ordinals
        """
        qX = self.quantize_input(X)
        # Check that self.fhe_tree is not None
        assert_true(
            self.fhe_tree is not None,
            f"You must call {self.compile.__name__} "
            f"before calling {self.predict.__name__} with execute_in_fhe=True.",
        )
        y_preds = numpy.zeros((qX.shape[0], self.n_classes_), dtype=numpy.int32)
        for i in range(qX.shape[0]):
            # FIXME transpose workaround see #292
            # expected x shape is (n_features, n_samples)
            fhe_pred = self.fhe_tree.run(qX[i].astype(numpy.uint8).reshape(qX[i].shape[0], 1))
            # Shape of y_pred is (n_trees, classes, n_examples)
            # For a single decision tree we can squeeze the first dimension
            # and get a shape of (classes, n_examples)
            fhe_pred = numpy.squeeze(fhe_pred, axis=0)
            y_preds[i, :] = fhe_pred.transpose()
        return y_preds

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

        qX = self.quantize_input(X)
        # _tensor_tree_predict needs X with shape (n_features, n_samples) but
        # X from sklearn is (n_samples, n_features)
        assert_true(
            qX.shape[1] == self.n_features_in_, "qX should have shape (n_samples, n_features)"
        )
        qX = qX.T
        y_pred = self._tensor_tree_predict(qX)[0]

        # Shape of y_pred is (n_trees, classes, n_examples)
        # For a single decision tree we can squeeze the first dimension
        # and get a shape of (classes, n_examples)
        y_pred = numpy.squeeze(y_pred, axis=0)

        # Transpose and reshape should be applied in clear.
        assert_true(
            (y_pred.shape[0] == self.n_classes_) and (y_pred.shape[1] == qX.shape[1]),
            "y_pred should have shape (n_classes, n_examples)",
        )
        y_pred = y_pred.transpose()
        return y_pred

    # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/365
    # add use_virtual_lib once the issue linked above is done
    def compile(
        self,
        X: numpy.ndarray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
    ):
        """Compile the model.

        Args:
            X (numpy.ndarray): the unquantized dataset
            compilation_configuration (Optional[CompilationConfiguration]): the options for
                compilation
            compilation_artifacts (Optional[CompilationArtifacts]): artifacts object to fill
                during compilation
            show_mlir (bool): whether or not to show MLIR during the compilation
            use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
                Defaults to False.

        """
        # Make sure that self.tree_predict is not None
        assert_true(
            self._tensor_tree_predict is not None, "You must fit the model before compiling it."
        )

        # mypy bug fix
        assert self._tensor_tree_predict is not None
        _tensor_tree_predict_proxy, parameters_mapping = generate_proxy_function(
            self._tensor_tree_predict, ["inputs"]
        )

        compiler_class = VirtualNPFHECompiler if use_virtual_lib else hnp.NPFHECompiler

        X = self.quantize_input(X)
        compiler = compiler_class(
            _tensor_tree_predict_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
            compilation_configuration,
            compilation_artifacts,
        )
        self.fhe_tree = compiler.compile_on_inputset(
            (sample.reshape(sample.shape[0], 1) for sample in X), show_mlir
        )
