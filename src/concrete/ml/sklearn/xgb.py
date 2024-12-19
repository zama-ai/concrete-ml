"""Implements XGBoost models."""

import platform
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy
import xgboost.sklearn
from numpy.random import RandomState
from xgboost.callback import TrainingCallback

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
        n_bits: Union[int, Dict[str, int]] = 6,
        max_depth: Optional[int] = 3,
        learning_rate: Optional[float] = None,
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
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        use_label_encoder: bool = False,
        random_state: Optional[int] = None,
        verbosity: Optional[int] = None,
        max_bin: Optional[int] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        early_stopping_rounds: Optional[int] = None,
        max_leaves: Optional[int] = None,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        max_cat_to_onehot: Optional[int] = None,
        grow_policy: Optional[str] = None,
        sampling_method: Optional[str] = None,
        **kwargs,
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
        self.max_bin = max_bin
        self.callbacks = callbacks
        self.early_stopping_rounds = early_stopping_rounds
        self.max_leaves = max_leaves
        self.eval_metric = eval_metric
        self.max_cat_to_onehot = max_cat_to_onehot
        self.grow_policy = grow_policy
        self.sampling_method = sampling_method
        self.kwargs = kwargs

    # pylint: disable=too-many-statements
    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["framework"] = self.framework
        metadata["post_processing_params"] = self.post_processing_params
        metadata["_fhe_ensembling"] = self._fhe_ensembling

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
        metadata["max_bin"] = self.max_bin
        metadata["early_stopping_rounds"] = self.early_stopping_rounds
        metadata["max_leaves"] = self.max_leaves
        metadata["max_cat_to_onehot"] = self.max_cat_to_onehot
        metadata["grow_policy"] = self.grow_policy
        metadata["sampling_method"] = self.sampling_method

        if callable(self.eval_metric):
            raise NotImplementedError("Callable eval_metric is not supported for serialization")

        if self.kwargs:
            raise NotImplementedError("kwargs are not supported for serialization")

        if self.callbacks:
            raise NotImplementedError("callbacks are not supported for serialization")

        metadata["eval_metric"] = self.eval_metric
        metadata["kwargs"] = None
        metadata["callbacks"] = None

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = XGBClassifier(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.framework = metadata["framework"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._fhe_ensembling = metadata["_fhe_ensembling"]

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4545
        # Execute with 2 example for efficiency in large data scenarios to prevent slowdown
        # but also to work around the HB export issue.
        obj._tree_inference = tree_to_numpy(
            obj.sklearn_model,
            numpy.tile(numpy.zeros((len(obj.input_quantizers),))[None, ...], [2, 1]),
            framework=obj.framework,
            output_n_bits=obj.n_bits["op_leaves"] if isinstance(obj.n_bits, Dict) else obj.n_bits,
            fhe_ensembling=obj._fhe_ensembling,
        )[0]
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
        obj.max_bin = metadata["max_bin"]
        obj.callbacks = metadata["callbacks"]
        obj.early_stopping_rounds = metadata["early_stopping_rounds"]
        obj.max_leaves = metadata["max_leaves"]
        obj.eval_metric = metadata["eval_metric"]
        obj.max_cat_to_onehot = metadata["max_cat_to_onehot"]
        obj.grow_policy = metadata["grow_policy"]
        obj.sampling_method = metadata["sampling_method"]

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
        n_bits: Union[int, Dict[str, int]] = 6,
        max_depth: Optional[int] = 3,
        learning_rate: Optional[float] = None,
        n_estimators: int = 20,
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
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        random_state: Optional[Union[RandomState, int]] = None,
        verbosity: Optional[int] = None,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        sampling_method: Optional[str] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        max_cat_to_onehot: Optional[int] = None,
        grow_policy: Optional[str] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any,
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
        self.random_state = random_state
        self.verbosity = verbosity
        self.max_bin = max_bin
        self.max_leaves = max_leaves
        self.max_cat_to_onehot = max_cat_to_onehot
        self.grow_policy = grow_policy
        self.sampling_method = sampling_method
        self.callbacks = callbacks
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.kwargs = kwargs

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

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        y_preds = super().post_processing(y_preds)

        # Hummingbird Gemm for XGBoostRegressor adds a + 0.5 at the end of the graph.
        # We need to add it back here since the graph is cut before this add node.
        y_preds += 0.5
        return y_preds

    # pylint: disable=too-many-statements
    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["framework"] = self.framework
        metadata["post_processing_params"] = self.post_processing_params
        metadata["_fhe_ensembling"] = self._fhe_ensembling

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
        metadata["random_state"] = self.random_state
        metadata["verbosity"] = self.verbosity
        metadata["max_bin"] = self.max_bin
        metadata["max_leaves"] = self.max_leaves
        metadata["max_cat_to_onehot"] = self.max_cat_to_onehot
        metadata["grow_policy"] = self.grow_policy
        metadata["sampling_method"] = self.sampling_method
        metadata["early_stopping_rounds"] = self.early_stopping_rounds

        if callable(self.eval_metric):
            raise NotImplementedError("Callable eval_metric is not supported for serialization")

        if self.kwargs:
            raise NotImplementedError("kwargs are not supported for serialization")

        if self.callbacks:
            raise NotImplementedError("callbacks are not supported for serialization")

        metadata["eval_metric"] = self.eval_metric
        metadata["kwargs"] = None
        metadata["callbacks"] = None

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = XGBRegressor(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.framework = metadata["framework"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._fhe_ensembling = metadata["_fhe_ensembling"]

        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4545
        # Execute with 2 example for efficiency in large data scenarios to prevent slowdown
        # but also to work around the HB export issue.
        obj._tree_inference = tree_to_numpy(
            obj.sklearn_model,
            numpy.tile(numpy.zeros((len(obj.input_quantizers),))[None, ...], [2, 1]),
            framework=obj.framework,
            output_n_bits=obj.n_bits["op_leaves"] if isinstance(obj.n_bits, Dict) else obj.n_bits,
            fhe_ensembling=obj._fhe_ensembling,
        )[0]
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
        obj.random_state = metadata["random_state"]
        obj.verbosity = metadata["verbosity"]
        obj.max_bin = metadata["max_bin"]
        obj.max_leaves = metadata["max_leaves"]
        obj.max_cat_to_onehot = metadata["max_cat_to_onehot"]
        obj.grow_policy = metadata["grow_policy"]
        obj.sampling_method = metadata["sampling_method"]
        obj.callbacks = metadata["callbacks"]
        obj.early_stopping_rounds = metadata["early_stopping_rounds"]
        obj.eval_metric = metadata["eval_metric"]
        obj.kwargs = metadata["kwargs"]

        return obj
