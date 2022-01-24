"""Implement the sklearn tree models."""
from __future__ import annotations

import copy
import warnings
from typing import Callable, Optional, Tuple

import concrete.numpy as hnp
import numpy
import onnx
import sklearn
from concrete.common.compilation.artifacts import CompilationArtifacts
from concrete.common.compilation.configuration import CompilationConfiguration
from concrete.common.fhe_circuit import FHECircuit
from onnx import numpy_helper

from ..common.debugging.custom_assert import assert_true
from ..common.utils import generate_proxy_function
from ..onnx.convert import get_equivalent_numpy_forward
from ..quantization.quantized_array import QuantizedArray

# pylint: disable=wrong-import-position,wrong-import-order

# Silence hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

N_BITS_ALLOWED = 8

# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name


class DecisionTreeClassifier(sklearn.tree.DecisionTreeClassifier):
    """Implements the sklearn DecisionTreeClassifier."""

    q_x_byfeatures: list
    fhe_tree: FHECircuit
    _tensor_tree_predict: Optional[Callable]

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
                as ``n_samples / (n_classes * np.bincount(y))``

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
        assert_true(n_bits <= N_BITS_ALLOWED, f"n_bits should be <= {N_BITS_ALLOWED}")
        self.n_bits = n_bits
        self._tensor_tree_predict = None
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
        # Deepcopy X as we don't want to alterate original values.
        X = copy.deepcopy(X)
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
            X[:, i] = q_x_.qvalues.astype(numpy.uint8)

        # Fit the model
        super().fit(X, y, *args, **kwargs)

        # Tree inference to numpy
        self._tensor_tree_predict = self.tree_to_numpy(X)

    def tree_to_numpy(self, X) -> Callable:
        """Convert the tree inference to a numpy functions using Hummingbird.

        Args:
            X (numpy.ndarray): The input data.

        Raises:
            ValueError: If onnx.graph.node input is not at the right position.

        Returns:
            Callable: A function that takes a numpy array and returns a numpy array.
        """
        # Silence hummingbird warnings
        warnings.filterwarnings("ignore")
        # TODO remove Transpose and Reshape from the list when (#292, #295) are done
        op_type_to_remove = ["Transpose", "ArgMax", "Reshape", "ReduceSum", "Cast"]

        # Convert model to onnx using hummingbird
        onnx_model = hb_convert(
            self, backend="onnx", test_input=X, extra_config={"tree_implementation": "gemm"}
        ).model

        op_type_inputs = {}
        # Replace not needed ops by Identity
        for node_index, node in enumerate(onnx_model.graph.node):
            # Save op_type for each node
            for output in node.output:
                op_type_inputs[output] = node.op_type
            if node.op_type in op_type_to_remove:
                # Check that node.input[0] is not a constant
                if node.input[0] != "input_0" and op_type_inputs[node.input[0]] == "Constant":
                    raise ValueError(
                        f"Trying to apply identity over a constant input." f"Node: {node.op_type}"
                    )  # pragma: no cover
                # Create a Identity node
                new_node = onnx.helper.make_node(
                    "Identity",
                    inputs=[str(node.input[0])],
                    outputs=node.output,
                )
                # Update current node with new_node
                onnx_model.graph.node[node_index].CopyFrom(new_node)

        # Modify onnx graph to fit in FHE
        for i, initializer in enumerate(onnx_model.graph.initializer):
            # Reshape initializer with shape (n_tree, hidden_size, n_features)
            # to (hidden_size, n_features). Concrete Numpy only accepts 2d matmul
            # TODO remove when 3d matmul is allowed (#293)
            if "weight_" in initializer.name and len(initializer.dims) == 3:
                onnx_model.graph.initializer[i].dims.pop(0)

            # All constants in our tree should be integers.
            # Tree thresholds can be rounded down (numpy.floor)
            # while the final probabilities (i.e. .predict())
            # has to take the value 1 for the class with highest probability.

            init_tensor = numpy_helper.to_array(initializer)
            if "weight_3" in initializer.name:
                # init_tensor should have only one 1 on axis=1
                # which correspond to max value on axis =1
                # (i.e. the class with highest probability)
                max_cols = numpy.argmax(init_tensor, axis=0)
                one_hot_classes = numpy.zeros(init_tensor.shape)
                for icol, m in enumerate(max_cols):
                    one_hot_classes[:, icol][m] = 1
                init_tensor = one_hot_classes
            else:
                init_tensor = numpy.floor(init_tensor)
            new_initializer = numpy_helper.from_array(init_tensor.astype(int), initializer.name)
            onnx_model.graph.initializer[i].CopyFrom(new_initializer)

        # Since the transpose is currently not implemented in concrete numpy
        # the input is transposed in clear. We need to update the Gemm
        # where the input is transposed.
        # FIXME remove this workaround once #292 is fixed

        # Find the Gemm node
        for node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "Gemm":
                gemm_node_index = node_index
                break

        # Gemm has transA and transB parameter. B is the input.
        # If we transpose the input before, we don't have to do it afterward.
        # In FHE we currently only send 1 example so the input looks has shape (1, n_features)
        # We simply need to transpose it to (n_features, 1)
        gemm_node = onnx_model.graph.node[gemm_node_index]
        new_node = numpy_helper.helper.make_node(
            name=gemm_node.name,
            op_type=gemm_node.op_type,
            inputs=gemm_node.input,
            outputs=gemm_node.output,
            alpha=1.0,
            beta=0.0,
        )
        onnx_model.graph.node[gemm_node_index].CopyFrom(new_node)
        _tensor_tree_predict = get_equivalent_numpy_forward(onnx_model)
        return _tensor_tree_predict

    def fit_benchmark(
        self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs
    ) -> Tuple[DecisionTreeClassifier, sklearn.tree.DecisionTreeClassifier]:
        """Fit the sklearn DecisionTreeClassifier and the FHE DecisionTreeClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            *args: args for super().fit
            **kwargs: kwargs for super().fit

        Returns:
            Tuple[DecisionTreeClassifier, sklearn.tree.DecisionTreeClassifier]:
                                                The FHE and sklearn DecisionTreeClassifier.
        """
        # Train the sklearn model without X quantized

        sklearn_model = sklearn.tree.DecisionTreeClassifier(**self.init_args)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model

    # pylint: disable=arguments-differ
    def predict(
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
        if execute_in_fhe:
            # Quantize the input
            X = self.quantize_input(X)
            # Check that self.fhe_tree is not None
            assert_true(
                self.fhe_tree is not None,
                f"You must call {self.compile.__name__} "
                f"before calling {self.predict.__name__} with execute_in_fhe=True.",
            )
            y_preds = numpy.zeros((X.shape[0], 2), dtype=numpy.int32)
            for i in range(X.shape[0]):
                fhe_pred = self.fhe_tree.run(X[i].astype(numpy.uint8).reshape(X[i].shape[0], 1))
                # Output has the shape (n_classes, n_examples).
                # Transpose to the predict sklearn like.
                y_preds[i, :] = fhe_pred.transpose()[0]
            # Convert to numpy array
            y_preds = y_preds[:, 1]
            return y_preds

        X = self._validate_X_predict(X, check_input)
        return self._predict_with_tensors(X)

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
        # _tensor_tree_predict needs X with shape (n_features, n_samples) but
        # X from sklearn is (n_samples, n_features)
        assert_true(
            X.shape[1] == self.n_features_in_, "X should have shape (n_samples, n_features)"
        )
        X = X.T
        y_pred = self._tensor_tree_predict(X)[0]

        # Shape of y_pred is (classes, n_examples)
        # Transpose and reshape should be applied in clear.
        assert_true(
            (y_pred.shape[0] == self.n_classes_) and (y_pred.shape[1] == X.shape[1]),
            "y_pred should have shape (n_classes, n_examples)",
        )
        y_pred = y_pred.transpose()
        return y_pred[:, 1]

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

    # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/365
    # add use_virtual_lib once the issue linked above is done
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

        # mypy bug fix
        assert self._tensor_tree_predict is not None
        _tensor_tree_predict_proxy, parameters_mapping = generate_proxy_function(
            self._tensor_tree_predict, ["inputs"]
        )

        X = self.quantize_input(X)
        compiler = hnp.NPFHECompiler(
            _tensor_tree_predict_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
            compilation_configuration,
            compilation_artifacts,
        )
        self.fhe_tree = compiler.compile_on_inputset(
            (sample.reshape(sample.shape[0], 1) for sample in X), show_mlir
        )
