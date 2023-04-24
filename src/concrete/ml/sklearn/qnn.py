"""Scikit-learn interface for fully-connected quantized neural networks."""

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name

import io
from typing import Any, Callable, Dict, Union

import numpy
import skorch.classifier
import skorch.regressor
import torch
from skorch.dataset import Dataset, ValidSplit
from torch.utils.data import DataLoader

from ..common.debugging import assert_true
from ..common.utils import FheMode, check_dtype_and_cast
from .base import QNN_AUTO_KWARGS, BaseClassifier, Data, QuantizedTorchEstimatorMixin, Target

# Define the QNN's support float and int dtypes
QNN_FLOAT_DTYPE = numpy.float32
QNN_INT_DTYPE = numpy.int64

# The different init parameters for the SparseQuantNeuralNetwork module
OPTIONAL_MODULE_PARAMS = [
    "n_hidden_neurons_multiplier",
    "n_w_bits",
    "n_a_bits",
    "n_accum_bits",
    "n_prune_neurons_percentage",
    "activation_function",
    "quant_narrow",
    "quant_signed",
]

# skorch's special attribute prefixes, which can be found in:
# https://skorch.readthedocs.io/en/v0.10.0/user/neuralnet.html#special-arguments
# Criterion and optimizer are handled separately using skorch's native `save_params` and
# `load_params` methods
ATTRIBUTE_PREFIXES = [
    "iterator_train",
    "iterator_valid",
    "callbacks",
    "dataset",
]


# We should also check that the `module__n_layers` parameter is properly set
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3553
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
                f"Setting `{auto_kwarg}` manually is forbidden. The number of inputs and outputs "
                "of the neural network are determined automatically in .fit, based on the data-set."
            )


# pylint: disable-next=too-many-instance-attributes
class NeuralNetRegressor(QuantizedTorchEstimatorMixin, skorch.regressor.NeuralNetRegressor):
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

    sklearn_model_class = skorch.regressor.NeuralNetRegressor
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

    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Save the model's weights/biases, optimizer and criterion attributes as well as their
        # related special arguments
        if self.sklearn_model is not None:

            # skorch's native `save_params` method dumps the objects into a file by default. In
            # order to avoid creating new files, we instead provide the method a buffer that we
            # then convert to a byte string and save it in the serialized json
            with io.BytesIO() as params, io.BytesIO() as optimizer, io.BytesIO() as criterion:

                # Make pruning permanent by removing weights associated to pruned neurons as Torch
                # does not allow to easily load and save pruned networks
                # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
                self.base_module.make_pruning_permanent()

                # We follow skorch's recommendation for saving and loading their models:
                # https://skorch.readthedocs.io/en/stable/user/save_load.html
                self.sklearn_model.save_params(
                    f_params=params,
                    f_optimizer=optimizer,
                    f_criterion=criterion,
                )

                metadata["params"] = params.getvalue().hex()
                metadata["optimizer"] = optimizer.getvalue().hex()
                metadata["criterion"] = criterion.getvalue().hex()

        # Concrete-ML
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["quantized_module_"] = self.quantized_module_
        metadata["post_processing_params"] = self.post_processing_params

        # skorch attributes that cannot be serialized
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3550
        # Disable mypy as running isinstance with a Callable type unexpectedly raises an issue:
        # https://github.com/python/mypy/issues/3060
        if isinstance(self.train_split, Callable) and not isinstance(  # type: ignore[arg-type]
            self.train_split, ValidSplit
        ):
            raise NotImplementedError(
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Please set `train_split` to either None or a ValidSplit instance."
            )

        if self.callbacks != "disable":
            raise NotImplementedError(
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Additionally, the serialization of skorch's different callback classes is not "
                f"supported. Please set `callbacks` to 'disable'. Got {self.callbacks}."
            )

        # Disable mypy as running isinstance with a Callable type unexpectedly raises an issue:
        # https://github.com/python/mypy/issues/3060
        if isinstance(self.predict_nonlinearity, Callable):  # type: ignore[arg-type]
            raise NotImplementedError(
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Please set`predict_nonlinearity` to either None or 'auto'."
            )

        # skorch
        metadata["lr"] = self.lr
        metadata["max_epochs"] = self.max_epochs
        metadata["batch_size"] = self.batch_size
        metadata["iterator_train"] = self.iterator_train
        metadata["iterator_valid"] = self.iterator_valid
        metadata["dataset"] = self.dataset
        metadata["train_split"] = self.train_split
        metadata["callbacks"] = self.callbacks
        metadata["predict_nonlinearity"] = self.predict_nonlinearity
        metadata["warm_start"] = self.warm_start
        metadata["verbose"] = self.verbose
        metadata["device"] = self.device
        metadata["history_"] = self.history_
        metadata["initialized_"] = self.initialized_
        metadata["virtual_params_"] = self.virtual_params_

        assert hasattr(
            self, "module__n_layers"
        ), f"{self.__class__.__name__} was not properly initialized."

        # skorch special argument (mandatory) for module : SparseQuantNeuralNetwork
        # pylint: disable-next=no-member
        metadata["module__n_layers"] = self.module__n_layers
        metadata["module__input_dim"] = self.module__input_dim
        metadata["module__n_outputs"] = self.module__n_outputs

        # skorch special argument (optional) for module : SparseQuantNeuralNetwork
        for module_param in OPTIONAL_MODULE_PARAMS:
            module_attribute = f"module__{module_param}"
            if hasattr(self, module_attribute):
                metadata[module_attribute] = getattr(self, module_attribute)

        # skorch special arguments
        # Coverage is disabled here as refactoring the serialization feature should remove this
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3250
        for attribute_prefix in ATTRIBUTE_PREFIXES:  # pragma: no cover
            for qnn_attribute in vars(self):
                if qnn_attribute.startswith(f"{attribute_prefix}__"):
                    metadata[qnn_attribute] = getattr(self, qnn_attribute)

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        # Instantiate the model
        obj = NeuralNetRegressor(
            module__n_layers=metadata["module__n_layers"],
        )

        # Concrete-ML
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj.quantized_module_ = metadata["quantized_module_"]
        obj.post_processing_params = metadata["post_processing_params"]

        # skorch
        obj.lr = metadata["lr"]
        obj.max_epochs = metadata["max_epochs"]
        obj.batch_size = metadata["batch_size"]
        obj.iterator_train = metadata["iterator_train"]
        obj.iterator_valid = metadata["iterator_valid"]
        obj.dataset = metadata["dataset"]
        obj.train_split = metadata["train_split"]
        obj.callbacks = metadata["callbacks"]
        obj.predict_nonlinearity = metadata["predict_nonlinearity"]
        obj.warm_start = metadata["warm_start"]
        obj.verbose = metadata["verbose"]
        obj.device = metadata["device"]
        obj.history_ = metadata["history_"]
        obj.initialized_ = metadata["initialized_"]
        obj.virtual_params_ = metadata["virtual_params_"]

        # skorch special argument (mandatory) for module : SparseQuantNeuralNetwork
        obj.module__input_dim = metadata["module__input_dim"]
        obj.module__n_outputs = metadata["module__n_outputs"]

        # skorch special argument (optional) for module : SparseQuantNeuralNetwork
        for module_param in OPTIONAL_MODULE_PARAMS:
            module_attribute = f"module__{module_param}"
            if module_attribute in metadata:
                setattr(obj, module_attribute, metadata[module_attribute])

        # skorch special arguments
        # Coverage is disabled here as refactoring the serialization feature should remove this
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3250
        for attribute_prefix in ATTRIBUTE_PREFIXES:  # pragma: no cover
            for qnn_attribute, qnn_value in metadata.items():
                if qnn_attribute.startswith(f"{attribute_prefix}__"):
                    setattr(obj, qnn_attribute, qnn_value)

        if "params" in metadata and "optimizer" in metadata and "criterion" in metadata:
            # Initialize the underlying model
            # We follow skorch's recommendation for saving and loading their models:
            # https://skorch.readthedocs.io/en/stable/user/save_load.html
            params = obj.get_sklearn_params()
            obj.sklearn_model = obj.sklearn_model_class(**params)
            obj.sklearn_model.initialize()

            # Make pruning permanent by removing weights associated to pruned neurons as Torch
            # does not allow to easily load and save pruned networks
            # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
            obj.base_module.make_pruning_permanent()

            # Load the model's weights/biases, optimizer and criterion attributes as well as their
            # related special arguments
            obj.sklearn_model.load_params(
                f_params=io.BytesIO(bytes.fromhex(metadata["params"])),
                f_optimizer=io.BytesIO(bytes.fromhex(metadata["optimizer"])),
                f_criterion=io.BytesIO(bytes.fromhex(metadata["criterion"])),
            )

        return obj


# pylint: disable-next=too-many-instance-attributes
class NeuralNetClassifier(
    BaseClassifier, QuantizedTorchEstimatorMixin, skorch.classifier.NeuralNetClassifier
):
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

    sklearn_model_class = skorch.classifier.NeuralNetClassifier
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

    def dump_dict(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        # Save the model's weights/biases, optimizer and criterion attributes as well as their
        # related special arguments
        if self.sklearn_model is not None:

            # skorch's native `save_params` method dumps the objects into a file by default. In
            # order to avoid creating new files, we instead provide the method a buffer that we
            # then convert to a byte string and save it in the serialized json
            with io.BytesIO() as params, io.BytesIO() as optimizer, io.BytesIO() as criterion:

                # Make pruning permanent by removing weights associated to pruned neurons as Torch
                # does not allow to easily load and save pruned networks
                # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
                self.base_module.make_pruning_permanent()

                # We follow skorch's recommendation for saving and loading their models:
                # https://skorch.readthedocs.io/en/stable/user/save_load.html
                self.sklearn_model.save_params(
                    f_params=params,
                    f_optimizer=optimizer,
                    f_criterion=criterion,
                )

                metadata["params"] = params.getvalue().hex()
                metadata["optimizer"] = optimizer.getvalue().hex()
                metadata["criterion"] = criterion.getvalue().hex()

        # Concrete-ML
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["quantized_module_"] = self.quantized_module_
        metadata["post_processing_params"] = self.post_processing_params

        # Classifier
        metadata["target_classes_"] = self.target_classes_
        metadata["n_classes_"] = self.n_classes_

        # skorch attributes that cannot be serialized
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3550
        # Disable mypy as running isinstance with a Callable type unexpectedly raises an issue:
        # https://github.com/python/mypy/issues/3060
        if isinstance(self.train_split, Callable) and not isinstance(  # type: ignore[arg-type]
            self.train_split, ValidSplit
        ):
            raise NotImplementedError(
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Please set `train_split` to either None or a ValidSplit instance."
            )

        if self.callbacks != "disable":
            raise NotImplementedError(
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Additionally, the serialization of skorch's different callback classes is not "
                f"supported. Please set `callbacks` to 'disable'. Got {self.callbacks}."
            )

        # Disable mypy as running isinstance with a Callable type unexpectedly raises an issue:
        # https://github.com/python/mypy/issues/3060
        if isinstance(self.predict_nonlinearity, Callable):  # type: ignore[arg-type]
            raise NotImplementedError(
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Please set`predict_nonlinearity` to either None or 'auto'."
            )

        # skorch
        metadata["lr"] = self.lr
        metadata["max_epochs"] = self.max_epochs
        metadata["batch_size"] = self.batch_size
        metadata["iterator_train"] = self.iterator_train
        metadata["iterator_valid"] = self.iterator_valid
        metadata["dataset"] = self.dataset
        metadata["train_split"] = self.train_split
        metadata["callbacks"] = self.callbacks
        metadata["predict_nonlinearity"] = self.predict_nonlinearity
        metadata["warm_start"] = self.warm_start
        metadata["verbose"] = self.verbose
        metadata["device"] = self.device
        metadata["history_"] = self.history_
        metadata["initialized_"] = self.initialized_
        metadata["virtual_params_"] = self.virtual_params_

        assert hasattr(
            self, "module__n_layers"
        ), f"{self.__class__.__name__} was not properly initialized."

        # skorch special argument (mandatory) for module : SparseQuantNeuralNetwork
        # pylint: disable-next=no-member
        metadata["module__n_layers"] = self.module__n_layers
        metadata["module__input_dim"] = self.module__input_dim
        metadata["module__n_outputs"] = self.module__n_outputs

        # skorch special argument (optional) for module : SparseQuantNeuralNetwork
        for module_param in OPTIONAL_MODULE_PARAMS:
            module_attribute = f"module__{module_param}"
            if hasattr(self, module_attribute):
                metadata[module_attribute] = getattr(self, module_attribute)

        # skorch special arguments
        # Coverage is disabled here as refactoring the serialization feature should remove this
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3250
        for attribute_prefix in ATTRIBUTE_PREFIXES:  # pragma: no cover
            for qnn_attribute in vars(self):
                if qnn_attribute.startswith(f"{attribute_prefix}__"):
                    metadata[qnn_attribute] = getattr(self, qnn_attribute)

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):
        # Instantiate the model
        obj = NeuralNetClassifier(
            module__n_layers=metadata["module__n_layers"],
        )

        # Concrete-ML
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj.quantized_module_ = metadata["quantized_module_"]
        obj.post_processing_params = metadata["post_processing_params"]

        # Classifier
        obj.target_classes_ = metadata["target_classes_"]
        obj.n_classes_ = metadata["n_classes_"]

        # skorch
        obj.lr = metadata["lr"]
        obj.max_epochs = metadata["max_epochs"]
        obj.batch_size = metadata["batch_size"]
        obj.iterator_train = metadata["iterator_train"]
        obj.iterator_valid = metadata["iterator_valid"]
        obj.dataset = metadata["dataset"]
        obj.train_split = metadata["train_split"]
        obj.callbacks = metadata["callbacks"]
        obj.predict_nonlinearity = metadata["predict_nonlinearity"]
        obj.warm_start = metadata["warm_start"]
        obj.verbose = metadata["verbose"]
        obj.device = metadata["device"]
        obj.history_ = metadata["history_"]
        obj.initialized_ = metadata["initialized_"]
        obj.virtual_params_ = metadata["virtual_params_"]

        # skorch special argument (mandatory) for module : SparseQuantNeuralNetwork
        obj.module__input_dim = metadata["module__input_dim"]
        obj.module__n_outputs = metadata["module__n_outputs"]

        # skorch special argument (optional) for module : SparseQuantNeuralNetwork
        for module_param in OPTIONAL_MODULE_PARAMS:
            module_attribute = f"module__{module_param}"
            if module_attribute in metadata:
                setattr(obj, module_attribute, metadata[module_attribute])

        # skorch special arguments
        # Coverage is disabled here as refactoring the serialization feature should remove this
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3250
        for attribute_prefix in ATTRIBUTE_PREFIXES:  # pragma: no cover
            for qnn_attribute, qnn_value in metadata.items():
                if qnn_attribute.startswith(f"{attribute_prefix}__"):
                    setattr(obj, qnn_attribute, qnn_value)

        if "params" in metadata and "optimizer" in metadata and "criterion" in metadata:
            # Initialize the underlying model
            # We follow skorch's recommendation for saving and loading their models:
            # https://skorch.readthedocs.io/en/stable/user/save_load.html
            params = obj.get_sklearn_params()
            obj.sklearn_model = obj.sklearn_model_class(**params)
            obj.sklearn_model.initialize()

            # Make pruning permanent by removing weights associated to pruned neurons as Torch
            # does not allow to easily load and save pruned networks
            # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
            obj.base_module.make_pruning_permanent()

            # Load the model's weights/biases, optimizer and criterion attributes as well as their
            # related special arguments
            obj.sklearn_model.load_params(
                f_params=io.BytesIO(bytes.fromhex(metadata["params"])),
                f_optimizer=io.BytesIO(bytes.fromhex(metadata["optimizer"])),
                f_criterion=io.BytesIO(bytes.fromhex(metadata["criterion"])),
            )

        return obj
