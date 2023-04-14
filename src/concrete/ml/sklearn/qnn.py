"""Scikit-learn interface for fully-connected quantized neural networks."""

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name

from typing import Any, Dict, Union

import numpy
import torch
from skorch.classifier import NeuralNetClassifier as SkorchNeuralNetClassifier
from skorch.dataset import Dataset, ValidSplit
from skorch.regressor import NeuralNetRegressor as SkorchNeuralNetRegressor
from torch.utils.data import DataLoader

from ..common.debugging import assert_true
from ..common.utils import FheMode, check_dtype_and_cast
from .base import QNN_AUTO_KWARGS, BaseClassifier, Data, QuantizedTorchEstimatorMixin, Target

# Define the QNN's support float and int dtypes
QNN_FLOAT_DTYPE = numpy.float32
QNN_INT_DTYPE = numpy.int64


def _check_qnn_kwargs(input_kwargs: Dict[str, Any]) -> None:
    """Check that a QNN model is not constructed with automatically computed parameters.

    Args:
        input_kwargs (dict): The keyword arguments to check.

    Raises:
        ValueError: If the automatically computed parameters are present in the keyword arguments.
    """

    if "n_bits" in input_kwargs:
        raise ValueError(
            "Setting `n_bits` in Quantized Neural Networks is not possible. Instead, initialize "
            "the model using `module__n_w_bits`, `module__n_a_bits` and `module__n_accum_bits` "
            "keyword arguments."
        )

    if "module" in input_kwargs:
        raise ValueError(
            "Setting `module` manually is forbidden. The module is set automatically when "
            "initializing the instance."
        )

    for auto_kwarg in QNN_AUTO_KWARGS:
        if auto_kwarg in input_kwargs:
            raise ValueError(
                f" Setting `{auto_kwarg}` manually is forbidden. The number of inputs and outputs "
                "of the neural network are determined automatically in .fit, based on the data-set."
            )


# QNNs do not support serialization yet
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3134
# pylint: disable-next=too-many-instance-attributes, abstract-method
class NeuralNetRegressor(QuantizedTorchEstimatorMixin, SkorchNeuralNetRegressor):
    """A Fully-Connected Neural Network regressor with FHE.

    This class wraps a quantized neural network implemented using Torch tools as a scikit-learn
    estimator. The skorch package allows to handle training and scikit-learn compatibility,
    and adds quantization as well as compilation functionalities. The neural network implemented
    by this class is a multi layer fully connected network trained with Quantization Aware Training
    (QAT).

    Inputs and targets that are float64 will be casted to float32 before training as Torch does not
    handle float64 types properly. Thus should not have a significant impact on the model's
    performances. An error is raised if these values are not floating points.
    """

    sklearn_model_class = SkorchNeuralNetRegressor
    _is_a_public_cml_model = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.01,
        max_epochs=10,
        batch_size=128,
        iterator_train=DataLoader,
        iterator_valid=DataLoader,
        dataset=Dataset,
        train_split=None,
        callbacks=None,
        predict_nonlinearity="auto",
        warm_start=False,
        verbose=1,
        device="cpu",
        **kwargs,
    ):
        # Call QuantizedTorchEstimatorMixin's __init__ method
        super().__init__()

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.iterator_train = iterator_train
        self.iterator_valid = iterator_valid
        self.dataset = dataset
        self.train_split = ValidSplit(5) if train_split is None else train_split
        self.callbacks = callbacks
        self.predict_nonlinearity = predict_nonlinearity
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device

        _check_qnn_kwargs(kwargs)

        history = kwargs.pop("history", None)
        initialized = kwargs.pop("initialized_", False)
        virtual_params = kwargs.pop("virtual_params_", {})

        self._kwargs = kwargs
        vars(self).update(kwargs)

        self.history_ = history
        self.initialized_ = initialized
        self.virtual_params_ = virtual_params

    def fit(self, X: Data, y: Target, *args, **kwargs):
        # Check that inputs and targets are float32. If they are float64, they will be casted to
        # float32 as this should not have a great impact on the model's performances. Else, an error
        # is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network regressor input")
        y = check_dtype_and_cast(y, "float32", error_information="Neural Network regressor target")

        # The number of outputs for regressions is the number of regression targets
        # We use y.shape which works for all supported datatype (including numpy array, pandas
        # dataframe and torch tensor).
        self.module__n_outputs = y.shape[1] if y.ndim == 2 else 1

        # Set the number of input dimensions to use
        self.module__input_dim = X.shape[1]

        # Call QuantizedTorchEstimatorMixin's fit method
        return super().fit(X, y, *args, **kwargs)

    def fit_benchmark(self, X: Data, y: Target, *args, **kwargs):
        # Check that inputs and targets are float32. If they are float64, they will be casted to
        # float32 as this should not have a great impact on the model's performances. Else, an error
        # is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network regressor input")
        y = check_dtype_and_cast(y, "float32", error_information="Neural Network regressor target")

        # Call QuantizedTorchEstimatorMixin's fit_benchmark method
        return super().fit_benchmark(X, y, *args, **kwargs)

    # skorch provides a predict_proba method for neural network regressors while scikit-learn does
    # not. We decided to follow scikit-learn's API as we build most of our tools on this library.
    # However, our models are still directly inheriting from skorch's classes, which makes this
    # method accessible by anyone, without having any FHE implementation. As this could create some
    # confusion, a NotImplementedError is raised. This issue could be fixed by making these classes
    # not inherit from skorch.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
    def predict_proba(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        raise NotImplementedError(
            "The `predict_proba` method is not implemented for neural network regressors. Please "
            "call `predict` instead."
        )

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Check that inputs are float32. If they are float64, they will be casted to float32 as
        # this should not have a great impact on the model's performances. Else, an error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network regressor input")

        # Call BaseEstimator's predict method and cast values to float32
        y_preds = super().predict(X, fhe=fhe)
        y_preds = self.post_processing(y_preds)
        return y_preds


# QNNs do not support serialization yet
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3134
# pylint: disable-next=too-many-instance-attributes, abstract-method
class NeuralNetClassifier(BaseClassifier, QuantizedTorchEstimatorMixin, SkorchNeuralNetClassifier):
    """A Fully-Connected Neural Network classifier with FHE.

    This class wraps a quantized neural network implemented using Torch tools as a scikit-learn
    estimator. The skorch package allows to handle training and scikit-learn compatibility,
    and adds quantization as well as compilation functionalities. The neural network implemented
    by this class is a multi layer fully connected network trained with Quantization Aware Training
    (QAT).

    Inputs that are float64 will be casted to float32 before training as Torch does not
    handle float64 types properly. Thus should not have a significant impact on the model's
    performances. If the targets are integers of lower bit-width, they will be safely casted to
    int64. Else, an error is raised.
    """

    sklearn_model_class = SkorchNeuralNetClassifier
    _is_a_public_cml_model = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        classes=None,
        lr=0.01,
        max_epochs=10,
        batch_size=128,
        iterator_train=DataLoader,
        iterator_valid=DataLoader,
        dataset=Dataset,
        train_split=None,
        callbacks=None,
        predict_nonlinearity="auto",
        warm_start=False,
        verbose=1,
        device="cpu",
        **kwargs,
    ):
        # Call BaseClassifier's __init__ method
        super().__init__()

        self.criterion = criterion
        self.optimizer = optimizer
        self.classes = classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.iterator_train = iterator_train
        self.iterator_valid = iterator_valid
        self.dataset = dataset
        self.train_split = ValidSplit(5, stratified=True) if train_split is None else train_split
        self.callbacks = callbacks
        self.predict_nonlinearity = predict_nonlinearity
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device

        _check_qnn_kwargs(kwargs)

        history = kwargs.pop("history", None)
        initialized = kwargs.pop("initialized_", False)
        virtual_params = kwargs.pop("virtual_params_", {})

        self._kwargs = kwargs
        vars(self).update(kwargs)

        self.history_ = history
        self.initialized_ = initialized
        self.virtual_params_ = virtual_params

    def fit(self, X: Data, y: Target, *args, **kwargs):
        # Check that inputs are float32 and targets are int64. If inputs are float64, they will be
        # casted to float32 as this should not have a great impact on the model's performances. If
        # the targets are integers of lower bit-width, they will be safely casted to int64. Else, an
        # error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network classifier input")
        y = check_dtype_and_cast(y, "int64", error_information="Neural Network classifier target")

        classes, y = numpy.unique(y, return_inverse=True)

        # Check that at least two classes are given
        n_classes = len(classes)
        assert_true(
            n_classes >= 2,
            f"Invalid number of classes: {str(n_classes)}, " "n_outputs should be larger than one",
        )

        # Set the number of outputs of the nn.Module to the number of classes
        self.module__n_outputs = n_classes

        # Set the number of input dimensions to use
        self.module__input_dim = X.shape[1]

        # Call BaseClassifier's fit method
        return super().fit(X, y, *args, **kwargs)

    def fit_benchmark(self, X: Data, y: Target, *args, **kwargs):
        # Check that inputs are float32 and targets are int64. If inputs are float64, they will be
        # casted to float32 as this should not have a great impact on the model's performances. If
        # the targets are integers of lower bit-width, they will be safely casted to int64. Else, an
        # error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network classifier input")
        y = check_dtype_and_cast(y, "int64", error_information="Neural Network classifier target")

        # Call QuantizedTorchEstimatorMixin's fit_benchmark method
        return super().fit_benchmark(X, y, *args, **kwargs)

    def predict_proba(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Check that inputs are float32. If they are, they will be casted to float32 as this
        # should not have a great impact on the model's performances. Else, an error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network classifier input")

        # Call BaseClassifier's predict_proba method, apply the sigmoid and cast values to float32
        y_logits = super().predict_proba(X, fhe=fhe)
        y_proba = self.post_processing(y_logits)
        return y_proba

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Check that inputs are float32. If they are float64, they will be casted to float32 as
        # this should not have a great impact on the model's performances. Else, an error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network classifier input")

        # Call BaseClassifier's predict method
        return super().predict(X, fhe=fhe)
