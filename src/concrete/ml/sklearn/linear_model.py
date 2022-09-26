"""Implement sklearn linear model."""
import copy
from typing import Any

import numpy
import sklearn
import sklearn.linear_model
import torch

from ..common.check_inputs import check_X_y_and_assert
from ..common.debugging.custom_assert import assert_true
from ..quantization import PostTrainingAffineQuantization
from ..torch import NumpyModule
from .base import SklearnLinearClassifierMixin, SklearnLinearModelMixin
from .torch_module import _CustomLinearRegressionTorchModel


# pylint: disable=invalid-name,too-many-instance-attributes
class LinearRegression(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A linear regression model with FHE.

    Arguments:
        n_bits(int): default is 2.
        use_sum_workaround (bool): indicate if the sum workaround should be used or not. This
            feature is experimental and should be used carefully. Important note: it only works
            for a LinearRegression model with N features, N a power of 2, for now. More
            information available in the QuantizedReduceSum operator. Default to False.

    For more details on LinearRegression please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    sklearn_alg = sklearn.linear_model.LinearRegression

    def __init__(
        self,
        n_bits=2,
        use_sum_workaround=False,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        # FIXME: Figure out how to add scikit-learn documentation into our object #893
        self.n_bits = n_bits
        self.use_sum_workaround = use_sum_workaround
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)

    # pylint: disable=attribute-defined-outside-init
    def fit(self, X, y: numpy.ndarray, *args, **kwargs) -> Any:
        """Fit the FHE linear model.

        Args:
            X : training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): The target data.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.

        Returns:
            Any
        """
        if not self.use_sum_workaround:
            return super().fit(X, y, *args, **kwargs)

        # If the ReduceSum experimental workaround should be used in order to enable summing many
        # integers without overflowing.

        input_shape = X.shape[1] if len(X.shape) > 1 else X.shape[0]
        target_number = y.shape[1] if len(y.shape) > 1 else 1

        assert_true(
            (input_shape != 0) and (input_shape & (input_shape - 1) == 0) and target_number == 1,
            "The sum workaround currently only handles N features with N a power of 2 and "
            f"single target values while {input_shape} features and {target_number} target(s)"
            "were given.",
        )

        # Copy X
        X = copy.deepcopy(X)

        # LinearRegression handles multi-labels data
        X, y = check_X_y_and_assert(X, y, multi_output=y.size > 1)

        # Initialize the sklearn model
        params = self.get_params()  # type: ignore
        params.pop("n_bits", None)
        params.pop("use_sum_workaround", None)
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(X, y, *args, **kwargs)

        # Extract the weights
        weights = torch.from_numpy(self.sklearn_model.coef_)
        bias = torch.tensor(self.sklearn_model.intercept_)

        # Initialize a Torch model that reproduces the proper inference using ReduceSum
        torch_model = _CustomLinearRegressionTorchModel(
            weights=weights,
            bias=bias,
        )

        # Create a NumpyModule from the Torch model
        numpy_module = NumpyModule(torch_model, dummy_input=torch.from_numpy(X[:1]))
        onnx_model = numpy_module.onnx_model

        self._onnx_model_ = onnx_model

        # Apply post-training quantization
        post_training = PostTrainingAffineQuantization(
            n_bits=self.n_bits, numpy_model=numpy_module, is_signed=True
        )

        # Calibrate and create quantize module
        self.quantized_module_ = post_training.quantize_module(X)

        return self


# pylint: enable=attribute-defined-outside-init


class ElasticNet(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """An ElasticNet regression model with FHE.

    Arguments:
        n_bits(int): default is 2.

    For more details on ElasticNet please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    """

    sklearn_alg = sklearn.linear_model.ElasticNet

    def __init__(
        self,
        n_bits=2,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        positive=False,
    ):
        # FIXME: Figure out how to add scikit-learn documentation into our object #893
        self.n_bits = n_bits
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)


class Lasso(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A Lasso regression model with FHE.

    Arguments:
        n_bits(int): default is 2.

    For more details on Lasso please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    sklearn_alg = sklearn.linear_model.Lasso

    def __init__(
        self,
        n_bits=2,
        alpha: float = 1.0,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        positive=False,
    ):
        # FIXME: Figure out how to add scikit-learn documentation into our object #893
        self.n_bits = n_bits
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)


class Ridge(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A Ridge regression model with FHE.

    Arguments:
        n_bits(int): default is 2.

    For more details on Ridge please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """

    sklearn_alg = sklearn.linear_model.Ridge

    def __init__(
        self,
        n_bits=2,
        alpha: float = 1.0,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        positive=False,
    ):
        # FIXME: Figure out how to add scikit-learn documentation into our object #893
        self.n_bits = n_bits
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)


class LogisticRegression(SklearnLinearClassifierMixin, sklearn.base.ClassifierMixin):
    """A logistic regression model with FHE.

    Arguments:
        n_bits(int): default is 2.

    For more details on LogisticRegression please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    sklearn_alg = sklearn.linear_model.LogisticRegression

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=2,
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
        # FIXME: Figure out how to add scikit-learn documentation into our object #893
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
        self._onnx_model_ = None
        super().__init__(n_bits=n_bits)


# pylint: enable=too-many-instance-attributes,invalid-name
