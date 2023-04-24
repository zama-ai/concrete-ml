"""Implements XGBoost models."""
import platform
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import xgboost.sklearn

from ..common.debugging.custom_assert import assert_true
from ..sklearn.tree_to_numpy import tree_to_numpy
from .base import BaseTreeClassifierMixin, BaseTreeRegressorMixin


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBClassifier(BaseTreeClassifierMixin):
    """Implements the XGBoost classifier.

    See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
    for more information about the parameters used.
    """

    sklearn_model_class = xgboost.sklearn.XGBClassifier
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
                warnings.warn(
                    "forcing n_jobs = 1 on mac for segfault issue", stacklevel=3
                )  # pragma: no cover
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

    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["framework"] = self.framework
        metadata["post_processing_params"] = self.post_processing_params

        # Classifier
        metadata["target_classes_"] = self.target_classes_
        metadata["n_classes_"] = self.n_classes_

        # XGBoost
        metadata["max_depth"] = self.max_depth
        metadata["learning_rate"] = self.learning_rate
        metadata["n_estimators"] = self.n_estimators
        metadata["objective"] = self.objective
        metadata["booster"] = self.booster
        metadata["tree_method"] = self.tree_method
        metadata["n_jobs"] = self.n_jobs
        metadata["gamma"] = self.gamma
        metadata["min_child_weight"] = self.min_child_weight
        metadata["max_delta_step"] = self.max_delta_step
        metadata["subsample"] = self.subsample
        metadata["colsample_bytree"] = self.colsample_bytree
        metadata["colsample_bylevel"] = self.colsample_bylevel
        metadata["colsample_bynode"] = self.colsample_bynode
        metadata["reg_alpha"] = self.reg_alpha
        metadata["reg_lambda"] = self.reg_lambda
        metadata["scale_pos_weight"] = self.scale_pos_weight
        metadata["base_score"] = self.base_score
        metadata["missing"] = self.missing
        metadata["num_parallel_tree"] = self.num_parallel_tree
        metadata["monotone_constraints"] = self.monotone_constraints
        metadata["interaction_constraints"] = self.interaction_constraints
        metadata["importance_type"] = self.importance_type
        metadata["gpu_id"] = self.gpu_id
        metadata["validate_parameters"] = self.validate_parameters
        metadata["predictor"] = self.predictor
        metadata["enable_categorical"] = self.enable_categorical
        metadata["use_label_encoder"] = self.use_label_encoder
        metadata["random_state"] = self.random_state
        metadata["verbosity"] = self.verbosity

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = XGBClassifier(n_bits=metadata["n_bits"])

        # Concrete-ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.framework = metadata["framework"]
        obj._tree_inference, obj.output_quantizers, obj.onnx_model_ = tree_to_numpy(
            obj.sklearn_model,
            numpy.zeros((len(obj.input_quantizers),))[None, ...],
            framework=obj.framework,
            output_n_bits=obj.n_bits,
        )
        obj.post_processing_params = metadata["post_processing_params"]

        # Classifier
        obj.target_classes_ = metadata["target_classes_"]
        obj.n_classes_ = metadata["n_classes_"]

        # XGBoost
        obj.max_depth = metadata["max_depth"]
        obj.learning_rate = metadata["learning_rate"]
        obj.n_estimators = metadata["n_estimators"]
        obj.objective = metadata["objective"]
        obj.booster = metadata["booster"]
        obj.tree_method = metadata["tree_method"]
        obj.n_jobs = metadata["n_jobs"]
        obj.gamma = metadata["gamma"]
        obj.min_child_weight = metadata["min_child_weight"]
        obj.max_delta_step = metadata["max_delta_step"]
        obj.subsample = metadata["subsample"]
        obj.colsample_bytree = metadata["colsample_bytree"]
        obj.colsample_bylevel = metadata["colsample_bylevel"]
        obj.colsample_bynode = metadata["colsample_bynode"]
        obj.reg_alpha = metadata["reg_alpha"]
        obj.reg_lambda = metadata["reg_lambda"]
        obj.scale_pos_weight = metadata["scale_pos_weight"]
        obj.base_score = metadata["base_score"]
        obj.missing = metadata["missing"]
        obj.num_parallel_tree = metadata["num_parallel_tree"]
        obj.monotone_constraints = metadata["monotone_constraints"]
        obj.interaction_constraints = metadata["interaction_constraints"]
        obj.importance_type = metadata["importance_type"]
        obj.gpu_id = metadata["gpu_id"]
        obj.validate_parameters = metadata["validate_parameters"]
        obj.predictor = metadata["predictor"]
        obj.enable_categorical = metadata["enable_categorical"]
        obj.use_label_encoder = metadata["use_label_encoder"]
        obj.random_state = metadata["random_state"]
        obj.verbosity = metadata["verbosity"]

        return obj


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name,too-many-instance-attributes
class XGBRegressor(BaseTreeRegressorMixin):
    """Implements the XGBoost regressor.

    See https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
    for more information about the parameters used.
    """

    sklearn_model_class = xgboost.sklearn.XGBRegressor
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
                warnings.warn(
                    "forcing n_jobs = 1 on mac for segfault issue", stacklevel=3
                )  # pragma: no cover
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

        # Hummingbird and XGBoost don't properly manage multi-outputs cases
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1856

        assert_true(
            (isinstance(y, list) and (not isinstance(y[0], list) or (len(y[0]) == 1)))
            or (not isinstance(y, list) and (len(y.shape) == 1 or y.shape[1] == 1)),
            "XGBRegressor doesn't support multi-output cases.",
        )

        # Call BaseTreeEstimatorMixin's fit method
        super().fit(X, y, *args, **kwargs)
        return self

    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["framework"] = self.framework
        metadata["post_processing_params"] = self.post_processing_params

        # XGBoost
        metadata["max_depth"] = self.max_depth
        metadata["learning_rate"] = self.learning_rate
        metadata["n_estimators"] = self.n_estimators
        metadata["objective"] = self.objective
        metadata["booster"] = self.booster
        metadata["tree_method"] = self.tree_method
        metadata["n_jobs"] = self.n_jobs
        metadata["gamma"] = self.gamma
        metadata["min_child_weight"] = self.min_child_weight
        metadata["max_delta_step"] = self.max_delta_step
        metadata["subsample"] = self.subsample
        metadata["colsample_bytree"] = self.colsample_bytree
        metadata["colsample_bylevel"] = self.colsample_bylevel
        metadata["colsample_bynode"] = self.colsample_bynode
        metadata["reg_alpha"] = self.reg_alpha
        metadata["reg_lambda"] = self.reg_lambda
        metadata["scale_pos_weight"] = self.scale_pos_weight
        metadata["base_score"] = self.base_score
        metadata["missing"] = self.missing
        metadata["num_parallel_tree"] = self.num_parallel_tree
        metadata["monotone_constraints"] = self.monotone_constraints
        metadata["interaction_constraints"] = self.interaction_constraints
        metadata["importance_type"] = self.importance_type
        metadata["gpu_id"] = self.gpu_id
        metadata["validate_parameters"] = self.validate_parameters
        metadata["predictor"] = self.predictor
        metadata["enable_categorical"] = self.enable_categorical
        metadata["use_label_encoder"] = self.use_label_encoder
        metadata["random_state"] = self.random_state
        metadata["verbosity"] = self.verbosity

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = XGBRegressor(n_bits=metadata["n_bits"])

        # Concrete-ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.framework = metadata["framework"]
        obj._tree_inference, obj.output_quantizers, obj.onnx_model_ = tree_to_numpy(
            obj.sklearn_model,
            numpy.zeros((len(obj.input_quantizers),))[None, ...],
            framework=obj.framework,
            output_n_bits=obj.n_bits,
        )
        obj.post_processing_params = metadata["post_processing_params"]

        # XGBoost
        obj.max_depth = metadata["max_depth"]
        obj.learning_rate = metadata["learning_rate"]
        obj.n_estimators = metadata["n_estimators"]
        obj.objective = metadata["objective"]
        obj.booster = metadata["booster"]
        obj.tree_method = metadata["tree_method"]
        obj.n_jobs = metadata["n_jobs"]
        obj.gamma = metadata["gamma"]
        obj.min_child_weight = metadata["min_child_weight"]
        obj.max_delta_step = metadata["max_delta_step"]
        obj.subsample = metadata["subsample"]
        obj.colsample_bytree = metadata["colsample_bytree"]
        obj.colsample_bylevel = metadata["colsample_bylevel"]
        obj.colsample_bynode = metadata["colsample_bynode"]
        obj.reg_alpha = metadata["reg_alpha"]
        obj.reg_lambda = metadata["reg_lambda"]
        obj.scale_pos_weight = metadata["scale_pos_weight"]
        obj.base_score = metadata["base_score"]
        obj.missing = metadata["missing"]
        obj.num_parallel_tree = metadata["num_parallel_tree"]
        obj.monotone_constraints = metadata["monotone_constraints"]
        obj.interaction_constraints = metadata["interaction_constraints"]
        obj.importance_type = metadata["importance_type"]
        obj.gpu_id = metadata["gpu_id"]
        obj.validate_parameters = metadata["validate_parameters"]
        obj.predictor = metadata["predictor"]
        obj.enable_categorical = metadata["enable_categorical"]
        obj.use_label_encoder = metadata["use_label_encoder"]
        obj.random_state = metadata["random_state"]
        obj.verbosity = metadata["verbosity"]

        return obj
