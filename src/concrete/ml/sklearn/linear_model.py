"""Implement sklearn linear model."""
import sklearn
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

    sklearn_alg = sklearn.linear_model.LinearRegression

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
        self.n_bits = n_bits
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        super().__init__(n_bits=n_bits)


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

    sklearn_alg = sklearn.linear_model.ElasticNet
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
        self.n_bits = n_bits
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
        super().__init__(n_bits=n_bits)


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

    sklearn_alg = sklearn.linear_model.Lasso
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
        self.n_bits = n_bits
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
        super().__init__(n_bits=n_bits)


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

    sklearn_alg = sklearn.linear_model.Ridge
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
        self.n_bits = n_bits
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state
        super().__init__(n_bits=n_bits)


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

    sklearn_alg = sklearn.linear_model.LogisticRegression
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
        super().__init__(n_bits=n_bits)


# pylint: enable=too-many-instance-attributes,invalid-name
