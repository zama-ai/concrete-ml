"""Implement the sklearn tree models."""
from typing import Callable, Optional

import numpy
import sklearn
from concrete.numpy.compilation.artifacts import CompilationArtifacts
from concrete.numpy.compilation.circuit import Circuit
from concrete.numpy.compilation.compiler import Compiler
from concrete.numpy.compilation.configuration import CompilationConfiguration
from custom_inherit import doc_inherit

from ..common.debugging.custom_assert import assert_true
from ..common.utils import generate_proxy_function
from ..quantization.quantized_array import QuantizedArray
from .base import BaseTreeEstimatorMixin
from .tree_to_numpy import tree_to_numpy

# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes


class DecisionTreeClassifier(
    BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator
):
    """Implements the sklearn DecisionTreeClassifier."""

    sklearn_alg = sklearn.tree.DecisionTreeClassifier
    q_x_byfeatures: list
    fhe_tree: Circuit
    _tensor_tree_predict: Optional[Callable]
    q_y: QuantizedArray
    class_mapping_: Optional[dict]
    n_classes_: int

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
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        BaseTreeEstimatorMixin.__init__(self, n_bits=n_bits)
        self.q_x_byfeatures = []
        self.n_bits = n_bits
        self.fhe_tree = None
        self.class_mapping_ = None

    # pylint: enable=too-many-arguments

    @doc_inherit(sklearn.tree.DecisionTreeClassifier.fit)
    def fit(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs):
        qX = numpy.zeros_like(X)

        self.q_x_byfeatures = []
        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i])
            self.q_x_byfeatures.append(q_x_)
            qX[:, i] = q_x_.qvalues.astype(numpy.int32)

        # If classes are not starting from 0 and/or increasing by 1
        # we need to map them to values 0, 1, ..., n_classes - 1
        classes_ = numpy.unique(y)
        if ~numpy.array_equal(numpy.arange(len(classes_)), classes_):
            self.class_mapping_ = dict(enumerate(classes_))

        # Register number of classes
        self.n_classes_ = len(classes_)

        # Initialize the model
        params = self.get_params()
        params.pop("n_bits", None)
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the model
        self.sklearn_model.fit(qX, y, *args, **kwargs)

        # Tree inference to numpy
        self._tensor_tree_predict, self.q_y = tree_to_numpy(
            self.sklearn_model,
            qX,
            "sklearn",
            output_n_bits=self.n_bits,
            use_workaround_for_transpose=True,
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

        # Make sure the shape of y_preds has 3 dimensions(n_tree, n_samples, n_classes)
        # and here n_tree = 1.
        assert_true(
            (y_preds.ndim == 3) and (y_preds.shape[0] == 1),
            f"Wrong dimensions for y_preds: {y_preds.shape} "
            f"when is should have shape (1, n_samples, n_classes)",
        )

        # Remove the first dimension in y_preds
        y_preds = y_preds[0]
        return y_preds

    # pylint: disable=arguments-differ
    @doc_inherit(sklearn.tree.DecisionTreeClassifier.predict_proba, style="google_with_merge")
    def predict_proba(
        self,
        X: numpy.ndarray,
        execute_in_fhe: Optional[bool] = False,
    ) -> numpy.ndarray:
        """Predict class probabilities of the input samples X.

        # noqa: DAR101

        Args:
            execute_in_fhe (bool, optional): If True, the predictions are computed in FHE.
                If False, the predictions are computed in the sklearn model. Defaults to False.

        Returns:
            numpy.ndarray: The class probabilities of the input samples X.

        """
        if execute_in_fhe:
            y_preds = self._execute_in_fhe(X)
        else:
            y_preds = self._predict_with_tensors(X)
        y_preds = self.post_processing(y_preds)
        return y_preds

    @doc_inherit(sklearn.tree.DecisionTreeClassifier.predict, style="google_with_merge")
    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict on user data.

        Predict on user data using either the quantized clear model,
        implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit

        # noqa: DAR101

        Args:
            execute_in_fhe (bool): whether to execute the inference in FHE

        Returns:
            the prediction as ordinals
        """
        y_preds = self.predict_proba(X, execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
        if self.class_mapping_ is not None:
            y_preds = numpy.array([self.class_mapping_[y_pred] for y_pred in y_preds])
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
        y_preds = numpy.zeros((1, qX.shape[0], self.n_classes_), dtype=numpy.int32)
        for i in range(qX.shape[0]):
            # FIXME transpose workaround see #292
            # expected x shape is (n_features, n_samples)
            fhe_pred = self.fhe_tree.encrypt_run_decrypt(
                qX[i].astype(numpy.uint8).reshape(qX[i].shape[0], 1)
            )
            y_preds[:, i, :] = numpy.transpose(fhe_pred, axes=(0, 2, 1))
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
            qX.shape[1] == self.sklearn_model.n_features_in_,
            "qX should have shape (n_samples, n_features)",
        )
        qX = qX.T
        y_pred = self._tensor_tree_predict(qX)[0]

        y_pred = numpy.transpose(y_pred, axes=(0, 2, 1))
        return y_pred

    def compile(
        self,
        X: numpy.ndarray,
        configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
    ):
        """Compile the model.

        Args:
            X (numpy.ndarray): the unquantized dataset
            configuration (Optional[CompilationConfiguration]): the options for
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

        X = self.quantize_input(X)
        compiler = Compiler(
            _tensor_tree_predict_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
            configuration,
            compilation_artifacts,
        )
        self.fhe_tree = compiler.compile(
            (sample.reshape(sample.shape[0], 1) for sample in X), show_mlir, virtual=use_virtual_lib
        )
