"""Implements XGBoost models."""
import platform
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import xgboost.sklearn

from ..common.debugging.custom_assert import assert_true
from .base import BaseTreeClassifierMixin, BaseTreeRegressorMixin


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBClassifier(BaseTreeClassifierMixin):
    """Implements the XGBoost classifier.

    See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
    for more information about the parameters used.
    """

    underlying_model_class = xgboost.sklearn.XGBClassifier
    framework = "xgboost"
    _is_a_public_cml_model = True

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
        random_state: Optional[int] = None,
        verbosity: Optional[int] = None,
    ):
        # base_score != 0.5 or None does not seem to not pass our tests
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/474
        assert_true(
            base_score in [0.5, None],
            f"Currently, only 0.5 or None are supported for base_score. Got {base_score}",
        )

        # See https://github.com/zama-ai/concrete-ml-internal/issues/503, there is currently
        # an issue with n_jobs != 1 on macOS
        #
        # When it gets fixed, we'll remove this workaround
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2747
        if platform.system() == "Darwin":
            if n_jobs != 1:  # pragma: no cover
                warnings.warn("forcing n_jobs = 1 on mac for segfault issue")  # pragma: no cover
                n_jobs = 1  # pragma: no cover

        # Call BaseClassifier's __init__ method
        super().__init__(n_bits=n_bits)

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


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBRegressor(BaseTreeRegressorMixin):
    """Implements the XGBoost regressor.

    See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
    for more information about the parameters used.
    """

    underlying_model_class = xgboost.sklearn.XGBRegressor
    framework = "xgboost"
    _is_a_public_cml_model = True

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
        random_state: Optional[int] = None,
        verbosity: Optional[int] = None,
    ):
        # base_score != 0.5 or None does not seem to not pass our tests
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/474
        assert_true(
            base_score in [0.5, None],
            f"Currently, only 0.5 or None are supported for base_score. Got {base_score}",
        )

        # See https://github.com/zama-ai/concrete-ml-internal/issues/503, there is currently
        # an issue with n_jobs != 1 on macOS
        #
        # When it gets fixed, we'll remove this workaround
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2747
        if platform.system() == "Darwin":
            if n_jobs != 1:  # pragma: no cover
                warnings.warn("forcing n_jobs = 1 on mac for segfault issue")  # pragma: no cover
                n_jobs = 1  # pragma: no cover

        # Call BaseTreeEstimatorMixin's __init__ method
        super().__init__(n_bits=n_bits)

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

    def fit(self, X, y, *args, **kwargs) -> Any:

        # HummingBird and XGBoost don't properly manage multi-outputs cases
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1856

        assert_true(
            (isinstance(y, list) and (not isinstance(y[0], list) or (len(y[0]) == 1)))
            or (not isinstance(y, list) and (len(y.shape) == 1 or y.shape[1] == 1)),
            "XGBRegressor doesn't support multi-output cases.",
        )

        # Call BaseTreeEstimatorMixin's fit method
        super().fit(X, y, *args, **kwargs)
        return self
