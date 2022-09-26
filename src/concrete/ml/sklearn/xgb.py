"""Implements XGBoost models."""
import platform
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
import xgboost.sklearn

from concrete.ml.quantization.quantizers import UniformQuantizer

from ..common.debugging.custom_assert import assert_true

# The sigmoid and softmax functions are already defined in the ONNX module and thus are imported
# here in order to avoid duplicating them.
from ..onnx.ops_impl import numpy_sigmoid, numpy_softmax
from ..quantization import QuantizedArray
from .base import BaseTreeClassifierMixin, BaseTreeRegressorMixin


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBClassifier(BaseTreeClassifierMixin):
    """Implements the XGBoost classifier."""

    sklearn_alg = xgboost.sklearn.XGBClassifier
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    output_quantizers: List[UniformQuantizer]
    _tensor_tree_predict: Optional[Callable]
    n_classes_: int
    sklearn_model: Any
    framework: str = "xgboost"

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        n_bits: int = 6,
        max_depth: Optional[int] = 3,
        learning_rate: Optional[float] = 0.1,
        n_estimators: Optional[int] = 20,
        objective: Optional[str] = "binary:logistic",
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = None,
        missing: float = numpy.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, List[Tuple[str]]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        use_label_encoder: bool = False,
        random_state: Optional[
            Union[numpy.random.RandomState, int]  # pylint: disable=no-member
        ] = None,
        verbosity: Optional[int] = None,
    ):
        # See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        # for more information about the parameters used.

        # base_score != 0.5 or None seems to not pass our tests (see #474)
        assert_true(
            base_score in [0.5, None],
            f"Currently, only 0.5 or None are supported for base_score. Got {base_score}",
        )

        # FIXME: see https://github.com/zama-ai/concrete-ml-internal/issues/503, there is currently
        # an issue with n_jobs != 1 on macOS
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/506, remove this workaround
        # once https://github.com/zama-ai/concrete-ml-internal/issues/503 is fixed
        if platform.system() == "Darwin":
            if n_jobs != 1:  # pragma: no cover
                warnings.warn("forcing n_jobs = 1 on mac for segfault issue")  # pragma: no cover
                n_jobs = 1  # pragma: no cover

        BaseTreeClassifierMixin.__init__(self, n_bits=n_bits)

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.predictor = predictor
        self.enable_categorical = enable_categorical
        self.use_label_encoder = use_label_encoder
        self.random_state = random_state
        self.verbosity = verbosity
        self.post_processing_params: Dict[str, Any] = {}

    def _update_post_processing_params(self):
        """Update the post processing params."""
        self.post_processing_params = {
            "n_classes_": self.n_classes_,
            "n_estimators": self.n_estimators,
        }

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        assert self.output_quantizers is not None

        # Update post-processing params with their current values
        self.__dict__.update(self.post_processing_params)

        y_preds = self.output_quantizers[0].dequant(y_preds)

        # Apply transpose
        y_preds = numpy.transpose(y_preds, axes=(2, 1, 0))

        # XGBoost returns a shape (n_examples, n_classes, n_trees) when self.n_classes_ >= 3
        # otherwise it returns a shape (n_examples, 1, n_trees)

        # Reshape to (-1, n_classes, n_trees)
        # No need to reshape if n_classes = 2
        if self.n_classes_ > 2:
            y_preds = y_preds.reshape((-1, self.n_classes_, self.n_estimators))  # type: ignore

        # Sum all tree outputs.
        y_preds = numpy.sum(y_preds, axis=2)
        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")

        # If this binary classification problem
        if self.n_classes_ == 2:
            # Apply sigmoid
            y_preds = numpy_sigmoid(y_preds)[0]

            # Transform in a 2d array where [1-p, p] is the output as XGBoost only outputs 1 value
            # when considering 2 classes
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)

        # Else, it's a multi-class classification problem
        else:
            # Apply softmax
            y_preds = numpy_softmax(y_preds)[0]

        return y_preds


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBRegressor(BaseTreeRegressorMixin):
    """Implements the XGBoost regressor."""

    sklearn_alg = xgboost.sklearn.XGBRegressor
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    output_quantizers: List[UniformQuantizer]
    _tensor_tree_predict: Optional[Callable]
    sklearn_model: Any
    framework: str = "xgboost"

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        n_bits: int = 6,
        max_depth: Optional[int] = 3,
        learning_rate: Optional[float] = 0.1,
        n_estimators: Optional[int] = 20,
        objective: Optional[str] = "reg:squarederror",
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = None,
        missing: float = numpy.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, List[Tuple[str]]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        use_label_encoder: bool = False,
        random_state: Optional[
            Union[numpy.random.RandomState, int]  # pylint: disable=no-member
        ] = None,
        verbosity: Optional[int] = None,
    ):
        # See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        # for more information about the parameters used.

        # base_score != 0.5 or None seems to not pass our tests (see #474)
        assert_true(
            base_score in [0.5, None],
            f"Currently, only 0.5 or None are supported for base_score. Got {base_score}",
        )

        # FIXME: see https://github.com/zama-ai/concrete-ml-internal/issues/503, there is currently
        # an issue with n_jobs != 1 on macOS
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/506, remove this workaround
        # once https://github.com/zama-ai/concrete-ml-internal/issues/503 is fixed
        if platform.system() == "Darwin":
            if n_jobs != 1:  # pragma: no cover
                warnings.warn("forcing n_jobs = 1 on mac for segfault issue")  # pragma: no cover
                n_jobs = 1  # pragma: no cover

        BaseTreeRegressorMixin.__init__(self, n_bits=n_bits)

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.predictor = predictor
        self.enable_categorical = enable_categorical
        self.use_label_encoder = use_label_encoder
        self.random_state = random_state
        self.verbosity = verbosity
        self.post_processing_params: Dict[str, Any] = {}

    def _update_post_processing_params(self):
        """Update the post processing params."""
        self.post_processing_params = {
            "n_estimators": self.n_estimators,
        }

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        assert self.output_quantizers is not None

        # Update post-processing params with their current values
        self.__dict__.update(self.post_processing_params)

        y_preds = self.output_quantizers[0].dequant(y_preds)

        # Apply transpose
        y_preds = numpy.transpose(y_preds, axes=(2, 1, 0))

        # XGBoost returns a shape (n_examples, n_classes, n_trees) when self.n_classes_ >= 3
        # otherwise it returns a shape (n_examples, 1, n_trees)

        # Sum all tree outputs.
        y_preds = numpy.sum(y_preds, axis=2)
        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")

        return y_preds

    def fit(self, X, y, **kwargs) -> Any:
        """Fit the tree-based estimator.

        Args:
            X : training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): The target data.
            **kwargs: args for super().fit

        Returns:
            Any: The fitted model.
        """

        # HummingBird doesn't manage correctly n_targets > 1
        assert_true(
            len(y.shape) == 1 or y.shape[1] == 1, "n_targets = 1 is the only supported case"
        )

        # Call super's fit that will train the network
        super().fit(X, y, **kwargs)
        return self
