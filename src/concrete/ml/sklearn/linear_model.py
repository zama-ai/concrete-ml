"""Implement sklearn linear model."""
from typing import Any, Dict

import sklearn.linear_model

from .base import SklearnLinearClassifierMixin, SklearnLinearRegressorMixin


# pylint: disable=invalid-name,too-many-instance-attributes
class LinearRegression(SklearnLinearRegressorMixin):
    """A linear regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LinearRegression please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    sklearn_model_class = sklearn.linear_model.LinearRegression

    _is_a_public_cml_model = True

    def __init__(
        self,
        n_bits=8,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # Scikit-Learn
        metadata["fit_intercept"] = self.fit_intercept
        metadata["normalize"] = self.normalize
        metadata["copy_X"] = self.copy_X
        metadata["n_jobs"] = self.n_jobs
        metadata["positive"] = self.positive

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = LinearRegression()

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.fit_intercept = metadata["fit_intercept"]
        obj.normalize = metadata["normalize"]
        obj.copy_X = metadata["copy_X"]
        obj.n_jobs = metadata["n_jobs"]
        obj.positive = metadata["positive"]
        return obj


class ElasticNet(SklearnLinearRegressorMixin):
    """An ElasticNet regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on ElasticNet please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    """

    sklearn_model_class = sklearn.linear_model.ElasticNet
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        normalize="deprecated",
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # Scikit-Learn
        metadata["alpha"] = self.alpha
        metadata["l1_ratio"] = self.l1_ratio
        metadata["fit_intercept"] = self.fit_intercept
        metadata["normalize"] = self.normalize
        metadata["copy_X"] = self.copy_X
        metadata["positive"] = self.positive
        metadata["precompute"] = self.precompute
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["warm_start"] = self.warm_start
        metadata["random_state"] = self.random_state
        metadata["selection"] = self.selection

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = ElasticNet()

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.alpha = metadata["alpha"]
        obj.l1_ratio = metadata["l1_ratio"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.normalize = metadata["normalize"]
        obj.copy_X = metadata["copy_X"]
        obj.positive = metadata["positive"]
        obj.precompute = metadata["precompute"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.warm_start = metadata["warm_start"]
        obj.random_state = metadata["random_state"]
        obj.selection = metadata["selection"]

        return obj


class Lasso(SklearnLinearRegressorMixin):
    """A Lasso regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on Lasso please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    sklearn_model_class = sklearn.linear_model.Lasso
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        alpha: float = 1.0,
        fit_intercept=True,
        normalize="deprecated",
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.selection = selection
        self.tol = tol
        self.precompute = precompute
        self.random_state = random_state

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # Scikit-Learn
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["normalize"] = self.normalize
        metadata["copy_X"] = self.copy_X
        metadata["positive"] = self.positive
        metadata["max_iter"] = self.max_iter
        metadata["warm_start"] = self.warm_start
        metadata["selection"] = self.selection
        metadata["tol"] = self.tol
        metadata["precompute"] = self.precompute
        metadata["random_state"] = self.random_state

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = Lasso()

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.normalize = metadata["normalize"]
        obj.copy_X = metadata["copy_X"]
        obj.positive = metadata["positive"]
        obj.max_iter = metadata["max_iter"]
        obj.warm_start = metadata["warm_start"]
        obj.selection = metadata["selection"]
        obj.tol = metadata["tol"]
        obj.precompute = metadata["precompute"]
        obj.random_state = metadata["random_state"]

        return obj


class Ridge(SklearnLinearRegressorMixin):
    """A Ridge regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on Ridge please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """

    sklearn_model_class = sklearn.linear_model.Ridge
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        alpha: float = 1.0,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        max_iter=None,
        tol=0.001,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # Scikit-Learn
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["normalize"] = self.normalize
        metadata["copy_X"] = self.copy_X
        metadata["positive"] = self.positive
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["solver"] = self.solver
        metadata["random_state"] = self.random_state

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = Ridge()

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Scikit-Learn
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.normalize = metadata["normalize"]
        obj.copy_X = metadata["copy_X"]
        obj.positive = metadata["positive"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.solver = metadata["solver"]
        obj.random_state = metadata["random_state"]
        return obj


class LogisticRegression(SklearnLinearClassifierMixin):
    """A logistic regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LogisticRegression please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    sklearn_model_class = sklearn.linear_model.LogisticRegression
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        # Call BaseClassifier's __init__ method
        super().__init__(n_bits=n_bits)

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete-ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # Classifier
        metadata["target_classes_"] = self.target_classes_
        metadata["n_classes_"] = self.n_classes_

        # Scikit-Learn
        metadata["penalty"] = self.penalty
        metadata["dual"] = self.dual
        metadata["tol"] = self.tol
        metadata["C"] = self.C
        metadata["fit_intercept"] = self.fit_intercept
        metadata["intercept_scaling"] = self.intercept_scaling
        metadata["class_weight"] = self.class_weight
        metadata["random_state"] = self.random_state
        metadata["solver"] = self.solver
        metadata["max_iter"] = self.max_iter
        metadata["multi_class"] = self.multi_class
        metadata["verbose"] = self.verbose
        metadata["warm_start"] = self.warm_start
        metadata["n_jobs"] = self.n_jobs
        metadata["l1_ratio"] = self.l1_ratio

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        # Instantiate the model
        obj = LogisticRegression()

        # Concrete-ML
        obj.n_bits = metadata["n_bits"]
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Classifier
        obj.target_classes_ = metadata["target_classes_"]
        obj.n_classes_ = metadata["n_classes_"]

        # Scikit-Learn
        obj.penalty = metadata["penalty"]
        obj.dual = metadata["dual"]
        obj.tol = metadata["tol"]
        obj.C = metadata["C"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.intercept_scaling = metadata["intercept_scaling"]
        obj.class_weight = metadata["class_weight"]
        obj.random_state = metadata["random_state"]
        obj.solver = metadata["solver"]
        obj.max_iter = metadata["max_iter"]
        obj.multi_class = metadata["multi_class"]
        obj.verbose = metadata["verbose"]
        obj.warm_start = metadata["warm_start"]
        obj.n_jobs = metadata["n_jobs"]
        obj.l1_ratio = metadata["l1_ratio"]

        return obj


# pylint: enable=too-many-instance-attributes,invalid-name
