"""Implements XGBoost models."""
import platform
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
import sklearn
import xgboost.sklearn

from concrete.ml.quantization.quantized_array import UniformQuantizer

from ..common.debugging.custom_assert import assert_true
from ..quantization import QuantizedArray
from .base import BaseTreeEstimatorMixin


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBClassifier(BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin):
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

        BaseTreeEstimatorMixin.__init__(self, n_bits=n_bits)

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

    def update_post_processing_params(self):
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

        if self.n_classes_ == 2:
            # Apply sigmoid (since xgboost output only 1 value when self.n_classes_ = 2
            y_preds = 1.0 / (1.0 + numpy.exp(-y_preds))
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
        else:
            # Otherwise we simply apply softmax
            y_preds = numpy.exp(y_preds)
            y_preds = y_preds / numpy.sum(y_preds, axis=1, keepdims=True)
        return y_preds
