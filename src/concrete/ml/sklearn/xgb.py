"""Implements XGBoost models."""
from __future__ import annotations

import platform
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy
import xgboost.sklearn

from ..common.debugging.custom_assert import assert_true
from ..quantization import QuantizedArray
from .base import BaseTreeEstimatorMixin
from .tree_to_numpy import tree_to_numpy


# Disabling invalid-name to use uppercase X
# pylint: disable=invalid-name
class XGBClassifier(xgboost.sklearn.XGBClassifier, BaseTreeEstimatorMixin):
    """Implements the XGBoost classifier."""

    sklearn_alg = xgboost.sklearn.XGBClassifier
    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    q_y: QuantizedArray
    _tensor_tree_predict: Optional[Callable]
    output_is_signed: bool

    CONCRETE_SPECIFIC_PARAMS: Set[str] = {"n_bits"}

    # pylint: disable=too-many-arguments,missing-docstring,too-many-locals
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
        **kwargs: Any,
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

        super_args = {**kwargs}
        eval_metric = super_args.pop("eval_metric", "logloss")

        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            gpu_id=gpu_id,
            validate_parameters=validate_parameters,
            predictor=predictor,
            enable_categorical=enable_categorical,
            use_label_encoder=use_label_encoder,
            random_state=random_state,
            verbosity=verbosity,
            eval_metric=eval_metric,
            **super_args,
        )
        BaseTreeEstimatorMixin.__init__(self, n_bits=n_bits)
        self.init_args = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "objective": objective,
            "booster": booster,
            "tree_method": tree_method,
            "n_jobs": n_jobs,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "max_delta_step": max_delta_step,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "base_score": base_score,
            "missing": missing,
            "num_parallel_tree": num_parallel_tree,
            "monotone_constraints": monotone_constraints,
            "interaction_constraints": interaction_constraints,
            "importance_type": importance_type,
            "gpu_id": gpu_id,
            "validate_parameters": validate_parameters,
            "predictor": predictor,
            "enable_categorical": enable_categorical,
            "use_label_encoder": use_label_encoder,
            "random_state": random_state,
            "verbosity": verbosity,
            **kwargs,
        }

    # pylint: enable=too-many-arguments,missing-docstring,too-many-locals

    #  pylint: disable=arguments-differ
    def fit(self, X: numpy.ndarray, y: numpy.ndarray, **kwargs) -> "XGBClassifier":
        """Fit the XGBoostClassifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            **kwargs: args for super().fit

        Returns:
            XGBoostClassifier: The XGBoostClassifier.
        """
        # mypy
        assert self.n_bits is not None

        qX = numpy.zeros_like(X)
        self.q_x_byfeatures = []

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i])
            self.q_x_byfeatures.append(q_x_)
            qX[:, i] = q_x_.qvalues.astype(numpy.int32)

        super().fit(qX, y, **kwargs)

        # Tree ensemble inference to numpy
        self._tensor_tree_predict, self.q_y = tree_to_numpy(
            self,
            qX,
            framework="xgboost",
            output_n_bits=self.n_bits,
            use_workaround_for_transpose=True,
        )
        return self

    def predict(
        self, X: numpy.ndarray, *args, execute_in_fhe: bool = False, **kwargs
    ) -> numpy.ndarray:
        """Predict the target values.

        Args:
            X (numpy.ndarray): The input data.
            args: args for super().predict
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.
            kwargs: kwargs for super().predict

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return BaseTreeEstimatorMixin.predict(
            self, X, *args, execute_in_fhe=execute_in_fhe, **kwargs
        )

    def predict_proba(
        self, X: numpy.ndarray, *args, execute_in_fhe: bool = False, **kwargs
    ) -> numpy.ndarray:
        """Predict the probabilities.

        Args:
            X (numpy.ndarray): The input data.
            args: args for super().predict
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.
            kwargs: kwargs for super().predict

        Returns:
            numpy.ndarray: The predicted probabilities.
        """
        return BaseTreeEstimatorMixin.predict_proba(
            self, X, *args, execute_in_fhe=execute_in_fhe, **kwargs
        )

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        y_preds = super().post_processing(y_preds)
        y_preds = 1.0 / (1.0 + numpy.exp(-y_preds))
        y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
        return y_preds

    def get_xgb_params(self) -> Dict[str, Any]:
        """Get xgboost specific parameters, removing concrete ml specific parameters.

        Returns:
            params (Dict): filtered parameters, eliminating parameters not used by XGB backend
        """
        params = super().get_xgb_params()
        # Parameters that should not go into native learner.
        filtered = {}
        for k, v in params.items():
            if k not in self.CONCRETE_SPECIFIC_PARAMS and not callable(v):
                filtered[k] = v
        return filtered

    def fit_benchmark(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        *args,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Fit the sklearn tree-based model and the FHE tree-based model.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            random_state (Optional[Union[int, numpy.random.RandomState, None]]):
                The random state. Defaults to None.
            *args: args for super().fit
            **kwargs: kwargs for super().fit

        Returns:
            Tuple[ConcreteEstimators, SklearnEstimators]:
                                                The FHE and sklearn tree-based models.
        """
        # Impose the evaluation metric unless the user requests an explicit one
        # This is passed to the original sklearn XGB classifier that is fitted by fit_benchmark
        kwargs["eval_metric"] = kwargs.get("eval_metric", "logloss")

        return super().fit_benchmark(X, y, *args, random_state=random_state, **kwargs)
