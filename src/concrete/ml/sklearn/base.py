"""Module that contains base classes for our libraries estimators."""
from __future__ import annotations

import functools
import json
import tempfile

# https://github.com/zama-ai/concrete-ml-internal/issues/942
# Refactoring base.py. This file is more than 1000 lines.
# We use names like X and q_X
# pylint: disable=too-many-lines,invalid-name
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import brevitas.nn as qnn
import numpy
import onnx
import sklearn
import torch
from brevitas.export.onnx.qonnx.manager import QONNXManager as BrevitasONNXManager
from concrete.numpy.compilation.artifacts import DebugArtifacts
from concrete.numpy.compilation.circuit import Circuit
from concrete.numpy.compilation.compiler import Compiler
from concrete.numpy.compilation.configuration import Configuration
from concrete.numpy.dtypes.integer import Integer

from concrete.ml.common.serialization import CustomEncoder
from concrete.ml.quantization.quantized_module import QuantizedModule, _get_inputset_generator
from concrete.ml.quantization.quantizers import QuantizationOptions, UniformQuantizer

from ..common.check_inputs import check_array_and_assert, check_X_y_and_assert_multi_output
from ..common.debugging.custom_assert import assert_true
from ..common.utils import (
    FheMode,
    check_there_is_no_p_error_options_in_configuration,
    generate_proxy_function,
    manage_parameters_for_pbs_errors,
)
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from ..onnx.onnx_model_manipulations import clean_graph_after_node_op_type, remove_node_types

# The sigmoid and softmax functions are already defined in the ONNX module and thus are imported
# here in order to avoid duplicating them.
from ..onnx.ops_impl import numpy_sigmoid, numpy_softmax
from ..quantization import PostTrainingQATImporter, QuantizedArray, get_n_bits_dict
from ..torch import NumpyModule
from .tree_to_numpy import tree_to_numpy

# Disable pylint to import hummingbird while ignoring the warnings
# pylint: disable=wrong-import-position,wrong-import-order
# Silence hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

_ALL_SKLEARN_MODELS: Set[Type] = set()
_LINEAR_MODELS: Set[Type] = set()
_TREE_MODELS: Set[Type] = set()
_NEURALNET_MODELS: Set[Type] = set()


class BaseEstimator:
    """Base class for all estimators in Concrete-ML.

    This class does not inherit from sklearn.base.BaseEstimator as it creates some conflicts
    with Skorch in QuantizedTorchEstimatorMixin's subclasses (more specifically, the `get_params`
    method is not properly inherited).

    Attributes:
        _is_a_public_cml_model (bool): Private attribute indicating if the class is a public model
            (as opposed to base or mixin classes).
    """

    #: Base float estimator class to consider for the model. Is set for each subclasses.
    sklearn_model_class: Any

    _is_a_public_cml_model: bool = False

    def __init__(self):
        """Initialize the base class with common attributes used in all estimators.

        An underscore "_" is appended to attributes that were created while fitting the model. This
        is done in order to follow Scikit-Learn's standard format. More information available
        in their documentation:
        https://scikit-learn.org/stable/developers/develop.html#:~:text=Estimated%20Attributes%C2%B6
        """
        #: The list of quantizers, which contain all information necessary for applying uniform
        #: quantization to inputs and provide quantization/dequantization functionalities. Is empty
        #: if the model is not fitted
        self.input_quantizers: List[UniformQuantizer] = []

        #: The list of quantizers, which contain all information necessary for applying uniform
        #: quantization to outputs and provide quantization/dequantization functionalities. Is
        #: empty if the model is not fitted
        self.output_quantizers: List[UniformQuantizer] = []

        #: The parameters needed for post-processing the outputs.
        #: Can be empty if no post-processing operations are needed for the associated model
        #: This attribute is typically used for serving models
        self.post_processing_params: Dict[str, Any] = {}

        #: Indicate if the model is fitted.
        self._is_fitted: bool = False

        #: Indicate if the model is compiled.
        self._is_compiled: bool = False

        self.fhe_circuit_: Optional[onnx.ModelProto] = None
        self.onnx_model_: Optional[Circuit] = None

    @property
    def onnx_model(self) -> Optional[onnx.ModelProto]:
        """Get the ONNX model.

        Is None if the model is not fitted.

        Returns:
            onnx.ModelProto: The ONNX model.
        """
        return self.onnx_model_

    @property
    def fhe_circuit(self) -> Optional[Circuit]:
        """Get the FHE circuit.

        The FHE circuit combines computational graph, mlir, client and server into a single object.
        More information available in Concrete-Numpy documentation:
        https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology
        Is None if the model is not fitted.

        Returns:
            Circuit: The FHE circuit.
        """
        return self.fhe_circuit_

    @fhe_circuit.setter
    def fhe_circuit(self, value: Circuit):
        """Set the FHE circuit.

        Args:
            value (Circuit): The FHE circuit to set.
        """
        self.fhe_circuit_ = value

    @property
    def is_fitted(self) -> bool:
        """Indicate if the model is fitted.

        Returns:
            bool: If the model is fitted.
        """
        return self._is_fitted

    def _is_not_fitted_error_message(self) -> str:
        return (
            f"The {self.__class__.__name__} model is not fitted. "
            "Please run fit(...) on proper arguments first."
        )

    def check_model_is_fitted(self):
        """Check if the model is fitted.

        Raises:
            AttributeError: If the model is not fitted.
        """
        if not self.is_fitted:
            raise AttributeError(self._is_not_fitted_error_message())

    def _is_not_compiled_error_message(self) -> str:
        return (
            f"The {self.__class__.__name__} model is not compiled. "
            "Please run compile(...) first before executing the prediction in FHE."
        )

    @property
    def is_compiled(self) -> bool:
        """Indicate if the model is compiled.

        Returns:
            bool: If the model is compiled.
        """
        return self._is_compiled

    def check_model_is_compiled(self):
        """Check if the model is compiled.

        Raises:
            AttributeError: If the model is not compiled.
        """
        if not self.is_compiled:
            raise AttributeError(self._is_not_compiled_error_message())

    def get_sklearn_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's
        parameters. It does not override Scikit-Learn's existing `get_params` method in order to
        not break its implementation of `set_params`.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained
                subobjects that are estimators. Default to True.

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        # Here, the `get_params` method is the `BaseEstimator.get_params` method from Scikit-Learn,
        # which will become available once a subclass inherits from it. We therefore disable both
        # pylint and mypy as this behavior is expected
        # pylint: disable-next=no-member
        params = super().get_params(deep=deep)  # type: ignore[misc]

        # Remove the n_bits parameters as this attribute is added by Concrete-ML
        params.pop("n_bits", None)

        return params

    def _set_post_processing_params(self):
        """Set parameters used in post-processing."""
        self.post_processing_params = {}

    def _fit_float_estimator(self, X, y, **fit_parameters) -> Any:
        """Fit the model's float equivalent estimator.

        Args:
            X: The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y: The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series
                or List.
            **fit_parameters: Keyword arguments to pass to the float estimator's fit method.

        Returns:
            Any: The fitted estimator.
        """
        # Retrieve the init parameters
        params = self.get_sklearn_params()

        # Initialize the sklearn model
        # pylint: disable-next=attribute-defined-outside-init
        self.sklearn_model = self.sklearn_model_class(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(X, y, **fit_parameters)

        return self.sklearn_model

    @abstractmethod
    def fit(self, X, y, **fit_parameters) -> Any:
        """Fit the estimator.

        This method trains a Scikit-Learn estimator, computes its ONNX graph and defines the
        quantization parameters needed for proper FHE inference.

        Args:
            X: The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y: The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series
                or List.
            **fit_parameters: Keyword arguments to pass to the float estimator's fit method.

        Returns:
            Any: The fitted estimator.
        """

    # Several attributes and methods are called in `fit_benchmark` but will only be accessible
    # in subclasses, we therefore need to disable pylint and mypy from checking these no-member
    # issues
    # pylint: disable=no-member
    def fit_benchmark(
        self,
        X,
        y,
        random_state: Optional[int] = None,
        **fit_parameters,
    ) -> Tuple[Any, Any]:
        """Fit both the Concrete-ML and its equivalent float estimators.

        Args:
            X: The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y: The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series
                or List.
            random_state (Optional[int]): The random state to use when fitting. Defaults to None.
            **fit_parameters: Keyword arguments to pass to the float estimator's fit method.

        Returns:
            Tuple[Any, Any]: The Concrete-ML and float equivalent fitted estimators.
        """

        # Retrieve sklearn's init parameters
        params = self.get_sklearn_params()

        # Make sure the random_state is set or both algorithms will diverge
        # due to randomness in the training.
        if "random_state" in params:
            if random_state is not None:
                params["random_state"] = random_state
            elif getattr(self, "random_state", None) is not None:
                # Disable mypy attribute definition errors as it does not seem to see that we make
                # sure this attribute actually exists before calling it
                params["random_state"] = self.random_state  # type: ignore[attr-defined]
            else:
                params["random_state"] = numpy.random.randint(0, 2**15)

        # Initialize the Scikit-Learn model
        sklearn_model = self.sklearn_model_class(**params)

        # Train the Scikit-Learn model
        sklearn_model.fit(X, y, **fit_parameters)

        # Update the Concrete-ML model's parameters
        # Disable mypy attribute definition errors as this attribute is expected to be
        # initialized once the model inherits from Skorch
        self.set_params(n_bits=self.n_bits, **params)  # type: ignore[attr-defined]

        # Train the Concrete-ML model
        self.fit(X, y, **fit_parameters)

        return self, sklearn_model

    # pylint: enable=no-member

    @abstractmethod
    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        """Quantize the input.

        This step ensures that the fit method has been called.

        Args:
            X (numpy.ndarray): The input values to quantize.

        Returns:
            numpy.ndarray: The quantized input values.
        """

    @abstractmethod
    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        """Dequantize the output.

        This step ensures that the fit method has been called.

        Args:
            q_y_preds (numpy.ndarray): The quantized output values to dequantize.

        Returns:
            numpy.ndarray: The dequantized output values.
        """

    @abstractmethod
    def _get_compiler(self) -> Compiler:
        """Retrieve the compiler instance to compile.

        Returns:
            Compiler: The compiler instance to compile.
        """

    def compile(
        self,
        X,
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose: bool = False,
    ) -> Circuit:
        """Compile the model.

        Args:
            X: A representative set of input values used for building cryptographic parameters.
            configuration (Optional[Configuration]): Options to use for compilation. Default
                to None.
            artifacts (Optional[DebugArtifacts]): Artifacts information about the
                compilation process to store for debugging.
            show_mlir (bool): Indicate if the MLIR graph should be printed during compilation.
            p_error (Optional[float]): Probability of error of a single PBS. A p_error value cannot
                be given if a global_p_error value is already set. Default to None, which sets this
                error to a default value.
            global_p_error (Optional[float]): Probability of error of the full circuit. A
                global_p_error value cannot be given if a p_error value is already set. This feature
                is not supported during Virtual Library simulation, meaning the probability is
                currently set to 0. Default to None, which sets this
                error to a default value.
            verbose (bool): Indicate if compilation information should be printed
                during compilation. Default to False.

        Returns:
            Circuit: The compiled Circuit.
        """
        # Reset for double compile
        self._is_compiled = False

        # Check that the model is correctly fitted
        self.check_model_is_fitted()

        # Cast pandas, list or torch to numpy
        X = check_array_and_assert(X)

        # p_error or global_p_error should not be set in both the configuration and direct arguments
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Find the right way to set parameters for compiler, depending on the way we want to default
        p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

        # Quantize the inputs
        q_X = self.quantize_input(X)

        # Generate the compilation inputset with proper dimensions
        inputset = _get_inputset_generator(q_X)

        # Retrieve the compiler instance
        module_to_compile = self._get_compiler()

        self.fhe_circuit = module_to_compile.compile(
            inputset,
            configuration=configuration,
            artifacts=artifacts,
            show_mlir=show_mlir,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose,
        )

        self._is_compiled = True

        return self.fhe_circuit

    @abstractmethod
    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        """Inference function to consider when executing in the clear.

        Args:
            q_X (numpy.ndarray): The quantized input values.

        Returns:
            numpy.ndarray: The quantized predicted values.
        """

    def predict(
        self, X: numpy.ndarray, fhe: Union[FheMode, str] = FheMode.DISABLE
    ) -> numpy.ndarray:
        """Predict values for X, in FHE or in the clear.

        Args:
            X (numpy.ndarray): The input values to predict.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete-ML python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            np.ndarray: The predicted values for X.
        """
        # Check that the model is properly fitted
        self.check_model_is_fitted()

        # Check that X's type and shape are supported
        X = check_array_and_assert(X)

        # Quantize the input
        q_X = self.quantize_input(X)

        # If the inference is executed in FHE or simulation mode
        if fhe in ["simulate", "execute"]:

            # Check that the model is properly compiled
            self.check_model_is_compiled()

            q_y_pred_list = []
            for q_X_i in q_X:
                # Expected encrypt_run_decrypt output shape is (1, n_features) while q_X_i
                # is of shape (n_features,)
                q_X_i = numpy.expand_dims(q_X_i, 0)

                # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3062
                # configuration.virtual is deprecated
                # Disable mypy's union attribute error as we already check that fhe_circuit is not
                # None using check_model_is_compiled
                q_y_pred_i = (
                    self.fhe_circuit.simulate(q_X_i)  # type: ignore[union-attr]
                    if fhe == "simulate"
                    else self.fhe_circuit.encrypt_run_decrypt(q_X_i)  # type: ignore[union-attr]
                )
                q_y_pred_list.append(q_y_pred_i[0])

            q_y_pred = numpy.array(q_y_pred_list)

        # Else, the prediction is simulated in the clear
        else:
            q_y_pred = self._inference(q_X)

        # Dequantize the predicted values in the clear
        y_pred = self.dequantize_output(q_y_pred)

        return y_pred

    # pylint: disable-next=no-self-use
    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the dequantized predictions.

        This post-processing step can include operations such as applying the sigmoid or softmax
        function for classifiers, or summing an ensemble's outputs. These steps are done in the
        clear because of current technical constraints. They most likely will be integrated in the
        FHE computations in the future.

        For some simple models such a linear regression, there is no post-processing step but the
        method is kept to make the API consistent for the client-server API. Other models might
        need to use attributes stored in `post_processing_params`.

        Args:
            y_preds (numpy.ndarray): The dequantized predictions to post-process.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        return y_preds

    @abstractmethod
    def dump_dict(self) -> Dict[str, Any]:
        """Dump the object as a dict.

        Returns:
            Dict[str, Any]: a dict representing the object
        """

    def dumps(self) -> str:
        """Dump itelf to a string.

        Returns:
            metadata (str): string of serialized object
        """
        metadata: Dict[str, Any] = self.dump_dict()
        return json.dumps(metadata, cls=CustomEncoder)

    def dump(self, file: IO[str]):
        """Dump itelf to a file.

        Args:
            file (IO[str]): file of where to dump.
        """
        metadata: Dict[str, Any] = self.dump_dict()
        json.dump(metadata, file, cls=CustomEncoder)

    @classmethod
    @abstractmethod
    def load_dict(cls, metadata: Dict[str, Any]) -> BaseEstimator:
        """Load itelf from a dict.

        Args:
            metadata (Dict[str, Any]): dict of metadata of the object

        Returns:
            BaseEstimator: the loaded object
        """

    @classmethod
    def load(cls, file: IO[str]) -> BaseEstimator:
        """Load itelf from a file.

        Args:
            file (IO[str]): file of serialized object

        Returns:
            BaseEstimator: the loaded object
        """
        metadata: Dict[str, Any] = json.load(file)
        return cls.load_dict(metadata=metadata)

    @classmethod
    def loads(cls, metadata: str) -> BaseEstimator:
        """Load itelf from a string.

        Args:
            metadata (str): serialized object

        Returns:
            BaseEstimator: the loaded object
        """
        _metadata: Dict = json.loads(metadata)
        return cls.load_dict(metadata=_metadata)


# This class only is an equivalent of BaseEstimator applied to classifiers, therefore not all
# methods are implemented and we need to disable pylint from checking that
# pylint: disable-next=abstract-method
class BaseClassifier(BaseEstimator):
    """Base class for linear and tree-based classifiers in Concrete-ML.

    This class inherits from BaseEstimator and modifies some of its methods in order to align them
    with classifier behaviors. This notably include applying a sigmoid/softmax post-processing to
    the predicted values as well as handling a mapping of classes in case they are not ordered.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #: The classifier's different target classes. Is None if the model is not fitted.
        self.classes_: Optional[numpy.ndarray] = None

        #: The classifier's number of different target classes. Is None if the model is not fitted.
        self.n_classes_: Optional[int] = None

    def _set_post_processing_params(self):
        super()._set_post_processing_params()
        self.post_processing_params.update({"n_classes_": self.n_classes_})

    def fit(self, X, y, **fit_parameters) -> Any:
        X, y = check_X_y_and_assert_multi_output(X, y)

        # Retrieve the different target classes
        classes = numpy.unique(y)
        self.classes_ = classes

        # Compute the number of target classes
        self.n_classes_ = len(classes)

        # Make sure y contains at least two classes
        assert_true(self.n_classes_ > 1, "You must provide at least 2 classes in y.")

        return super().fit(X, y, **fit_parameters)

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3261
    def predict_proba(self, X, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        """Predict class probabilities.

        Args:
            X: The input values to predict.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete-ML python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return super().predict(X, fhe=fhe)

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3261
    def predict(self, X, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Compute the predicted probabilities
        y_preds = self.predict_proba(X, fhe=fhe)

        # Retrieve the class with the highest probability
        y_preds = numpy.argmax(y_preds, axis=1)

        assert self.classes_ is not None, self._is_not_fitted_error_message()

        return self.classes_[y_preds]

    def post_processing(self, y_preds: numpy.ndarray):
        y_preds = super().post_processing(y_preds)

        # Retrieve the number of target classes
        n_classes_ = self.post_processing_params["n_classes_"]

        # If the predictions only has one dimension (i.e. binary classification problem), apply the
        # sigmoid operator
        if n_classes_ == 2:
            y_preds = numpy_sigmoid(y_preds)[0]

            # Transform in a 2d array where [1-p, p] is the output as scikit-learn only outputs 1
            # value when considering 2 classes
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)

        # Else, apply the softmax operator
        else:
            y_preds = numpy_softmax(y_preds)[0]

        return y_preds


class QuantizedTorchEstimatorMixin(BaseEstimator):
    """Mixin that provides quantization for a torch module and follows the Estimator API."""

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if getattr(klass, "_is_a_public_cml_model", False):
                _NEURALNET_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(
        self,
    ):
        self.quantized_module_ = QuantizedModule()
        super().__init__()

    @property
    @abstractmethod
    def base_module_to_compile(self):
        """Get the Torch module that should be compiled to FHE."""

    @property
    def input_quantizers(self) -> List[UniformQuantizer]:
        """Get the input quantizers.

        Returns:
            List[UniformQuantizer]: The input quantizers.
        """
        return self.quantized_module_.input_quantizers

    @input_quantizers.setter
    def input_quantizers(self, value: List[UniformQuantizer]):
        self.quantized_module_.input_quantizers = value

    @property
    def output_quantizers(self) -> List[UniformQuantizer]:
        """Get the output quantizers.

        Returns:
            List[UniformQuantizer]: The output quantizers.
        """
        return self.quantized_module_.output_quantizers

    @output_quantizers.setter
    def output_quantizers(self, value: List[UniformQuantizer]):
        self.quantized_module_.output_quantizers = value

    @property
    def fhe_circuit(self) -> Circuit:
        return self.quantized_module_.fhe_circuit

    @fhe_circuit.setter
    def fhe_circuit(self, value: Circuit):
        self.quantized_module_.fhe_circuit = value

    def _set_post_processing_params(self):
        # Disable mypy attribute definition error as this method is expected to be reachable
        # once the model inherits from Skorch
        params = self._get_predict_nonlinearity()  # type: ignore[attr-defined]

        if isinstance(params, functools.partial):
            post_processing_function_name = params.func.__name__
            post_processing_function_keywords = params.keywords
        else:
            post_processing_function_name = params.__name__
            post_processing_function_keywords = {}
        post_processing_params: Dict[str, Any] = {}
        post_processing_params["post_processing_function_name"] = post_processing_function_name
        post_processing_params[
            "post_processing_function_keywords"
        ] = post_processing_function_keywords
        self.post_processing_params = post_processing_params

    def _fit_float_estimator(self, X, y, **fit_parameters) -> Any:
        # Call Skorch's fit that will train the network. This will instantiate the model class if
        # it's not already done
        return self.sklearn_model_class.fit(self, X, y, **fit_parameters)

    def fit(self, X, y, **fit_parameters) -> Any:
        """Fit he estimator.

        If the module was already initialized, the module will be re-initialized unless
        `warm_start` is set to True. In addition to the torch training step, this method performs
        quantization of the trained Torch model.

        Values of dtype float64 are not supported and will be casted to float32.

        Args:
            X: The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y: The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series
                or List.
            **fit_parameters: Keyword arguments to pass to Skorch's fit method.

        Returns:
            Any: The fitted estimator.
        """
        # Reset for double fit
        self._is_fitted = False

        # Reset the quantized module since quantization is lost during refit
        # This will make the .infer() function call into the Torch nn.Module
        # Instead of the quantized module
        self.quantized_module_ = QuantizedModule()

        X, y = check_X_y_and_assert_multi_output(X, y)

        # A helper for users so they don't need to import torch directly
        args_to_convert_to_tensor = ["criterion__weight"]
        for arg_name in args_to_convert_to_tensor:
            if hasattr(self, arg_name):
                attr_value = getattr(self, arg_name)
                # The parameter could be a list, numpy.ndaray or tensor
                if isinstance(attr_value, list):
                    attr_value = numpy.asarray(attr_value, numpy.float32)

                if isinstance(attr_value, numpy.ndarray):
                    setattr(self, arg_name, torch.from_numpy(attr_value).float())

                assert_true(
                    isinstance(getattr(self, arg_name), torch.Tensor),
                    f"NeuralNetClassifier parameter `{arg_name}` must "
                    "be a numpy.ndarray, list or torch.Tensor",
                )

        # Fit the model by using Skorch's fit
        self._fit_float_estimator(X, y, **fit_parameters)

        # Export the brevitas model to ONNX
        output_onnx_file_path = Path(tempfile.mkstemp(suffix=".onnx")[1])

        BrevitasONNXManager.export(
            self.base_module_to_compile,
            input_shape=X[[0], ::].shape,
            export_path=str(output_onnx_file_path),
            keep_initializers_as_inputs=False,
            opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
        )

        onnx_model = onnx.load(str(output_onnx_file_path))

        output_onnx_file_path.unlink()

        # Create corresponding numpy model
        numpy_model = NumpyModule(onnx_model, torch.tensor(X[[0], ::]))

        self.onnx_model_ = numpy_model.onnx_model

        # Set the quantization bits for import
        # Note that the ONNXConverter will use a default value for network input bits
        # Furthermore, Brevitas ONNX contains bitwidths in the ONNX file
        # which override the bitwidth that we pass here
        # Thus, this parameter, set by the inheriting classes, such as NeuralNetClassifier
        # is only used to check consistency during import (onnx file vs import)
        # Disable mypy attribute definition error as this attribute is expected to be
        # initialized once the model inherits from Skorch
        n_bits = self.n_bits_quant  # type: ignore[attr-defined]

        # Import the quantization aware trained model
        qat_model = PostTrainingQATImporter(n_bits, numpy_model)

        self.quantized_module_ = qat_model.quantize_module(X)

        # Set post-processing params
        self._set_post_processing_params()

        self._is_fitted = True

        return self

    def _get_equivalent_float_module(self):
        """Build a topologically equivalent Torch module that can be used on floating points.

        Returns:
            float_module (nn.Sequential): The equivalent float module.
        """
        # Instantiate a new sequential module
        float_module = torch.nn.Sequential()

        layer_index = -1

        # Iterate over the model's sub-modules
        for module in self.base_module_to_compile.features:

            # If the module is not a QuantIdentity, it's either a QuantLinear or an activation
            if not isinstance(module, qnn.QuantIdentity):

                # If the module is a QuantLinear, replace it with a Linear module
                if isinstance(module, qnn.QuantLinear):
                    layer_index += 1

                    linear_name = f"fc{layer_index}"
                    linear_layer = torch.nn.Linear(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                    )

                    float_module.add_module(linear_name, linear_layer)

                # Else, it's a module representing the activation function, which needs to be
                # added as well
                else:
                    activation_name = f"act{layer_index}"
                    float_module.add_module(activation_name, module)

        return float_module

    # Skorch's Neural Networks don't support a random_state as parameter, we therefore need to
    # disable pylint as the BaseEstimator class expect the fit method to support it.
    def fit_benchmark(
        self, X, y, random_state: Optional[int] = None, **fit_parameters
    ) -> Tuple[Any, Any]:
        """Fit the quantized estimator as well as its equivalent float estimator.

        This function returns both the quantized estimator (itself) as well as its non-quantized
        (float) equivalent, which are both trained separately. This method differs from the
        BaseEstimator's `fit_benchmark` method as QNNs use QAT instead of PTQ. Hence, here, the
        float model is topologically equivalent as we have less control over the influence of QAT
        over the weights.

        Values of dtype float64 are not supported and will be casted to float32.

        Args:
            X: The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y: The target data,  as a Numpy array, Torch tensor, Pandas DataFrame Pandas Series
                or List.
            random_state (Optional[int]): The random state to use when fitting. However, Skorch
                does not handle such a parameter and setting it will have no effect. Defaults
                to None.
            **fit_parameters: Keyword arguments to pass to Skorch's fit method.

        Returns:
            Tuple[Any, Any]: The Concrete-ML and equivalent Skorch fitted estimators.
        """

        assert (
            random_state is None
        ), "Neural Network models do not support random_state as a parameter when fitting."

        # Fit the quantized estimator
        self.fit(X, y, **fit_parameters)

        # Retrieve the Skorch estimator's training parameters
        estimator_parameters = self.get_sklearn_params()

        # Retrieve the estimator's float topological equivalent module
        float_module = self._get_equivalent_float_module()

        # Instantiate the float estimator
        skorch_estimator_type = self.sklearn_model_class
        float_estimator = skorch_estimator_type(float_module, **estimator_parameters)

        # Fit the float estimator
        float_estimator.fit(X, y, **fit_parameters)

        return self, float_estimator

    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        q_X = self.quantized_module_.quantize_input(X)

        assert numpy.array(q_X).dtype == numpy.int64, "Inputs were not quantized to int64 values"
        return q_X

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        return self.quantized_module_.dequantize_output(q_y_preds)

    def compile(self, X, *args, **kwargs) -> Circuit:
        # Reset for double compile
        self._is_compiled = False

        # Check that the model is correctly fitted
        self.check_model_is_fitted()

        # Cast pandas, list or torch to numpy
        X = check_array_and_assert(X)

        # Compile the QuantizedModule
        # The QuantizedModule's compile method does not exactly share the same parameters as the
        # one from a Compiler instance. We therefore need to override the BaseEstimator's compile
        # method for that particular reason
        self.quantized_module_.compile(X, *args, **kwargs)

        # Make sure that no avoidable TLUs are found in the built-in model
        succ = list(self.fhe_circuit.graph.graph.successors(self.fhe_circuit.graph.input_nodes[0]))
        assert_true(
            not any(s.converted_to_table_lookup for s in succ),
            "The compiled circuit currently applies some lookup tables on input nodes but this "
            "should be avoided. Please check the underlying nn.Module.",
        )

        self._is_compiled = True

        return self.fhe_circuit

    def predict(self, X, fhe: Union[FheMode, str] = FheMode.DISABLE):
        return self.predict_proba(X, fhe=fhe)

    def predict_proba(self, X, fhe: Union[FheMode, str] = FheMode.DISABLE):
        """Predict values for a regressor and class probabilities for a classifier.

        In Scikit-Learn, predict_proba is not defined for regressors. However, Skorch seems to use
        it as `predict`, and defines what to return using a post-processing method. We therefore
        decided to follow the same structure and therefore include `predict_proba` in the QNN's
        base class.

        Values of dtype float64 are not supported and will be casted to float32.

        Args:
            X: The input values to predict.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete-ML python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            numpy.ndarray: The predicted values or class probabilities.
        """
        if fhe == "execute":
            # Run over each element of X individually and aggregate predictions in a vector
            if X.ndim == 1:
                X = X.reshape((1, -1))

            return super().predict(X, fhe="execute")

        # For prediction in the clear, we call the  Skorch's NeuralNet `predict_proba method which
        # ends up calling `infer`
        return self.sklearn_model_class.predict_proba(self, X).astype(numpy.float32)

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        if self.post_processing_params["post_processing_function_name"] == "softmax":
            # Get dim argument
            dim = self.post_processing_params["post_processing_function_keywords"]["dim"]

            # Apply softmax to the output
            y_preds = numpy_softmax(y_preds, axis=dim)[0]

        elif "sigmoid" in self.post_processing_params["post_processing_function_name"]:
            # Transform in a 2d array where [p, 1-p] is the output
            y_preds = numpy.concatenate((y_preds, 1 - y_preds), axis=1)  # pragma: no cover

        elif self.post_processing_params["post_processing_function_name"] == "_identity":
            pass

        else:
            raise ValueError(
                "Unknown post-processing function "
                f"{self.post_processing_params['post_processing_function_name']}"
            )  # pragma: no cover

        # To match torch softmax we need to cast to float32
        return y_preds.astype(numpy.float32)


class BaseTreeEstimatorMixin(BaseEstimator, sklearn.base.BaseEstimator, ABC):
    """Mixin class for tree-based estimators.

    This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's
    `get_params` and `set_params` methods.
    """

    #: Model's base framework used, either 'xgboost' or 'sklearn'. Is set for each subclasses.
    framework: str

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if getattr(klass, "_is_a_public_cml_model", False):
                _TREE_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, n_bits: int):
        """Initialize the TreeBasedEstimatorMixin.

        Args:
            n_bits (int): The number of bits used for quantization.
        """
        self.n_bits: int = n_bits

        #: The equivalent fitted float model. Is None if the model is not fitted.
        self.sklearn_model: Optional[Any] = None

        #: The model's inference function. Is None if the model is not fitted.
        self._tree_inference: Optional[Callable] = None

        BaseEstimator.__init__(self)

    def _sklearn_model_is_not_fitted_error_message(self) -> str:
        return (
            f"The underlying model (class: {self.sklearn_model_class}) is not fitted and thus "
            "cannot be quantized."
        )  # pragma: no cover

    def fit(self, X, y, **fit_parameters) -> Any:
        # Reset for double fit
        self._is_fitted = False

        X, y = check_X_y_and_assert_multi_output(X, y)

        q_X = numpy.zeros_like(X)
        self.input_quantizers = []

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            input_quantizer = QuantizedArray(n_bits=self.n_bits, values=X[:, i]).quantizer
            self.input_quantizers.append(input_quantizer)
            q_X[:, i] = input_quantizer.quant(X[:, i])

        # Fit the Scikit-Learn model
        self._fit_float_estimator(q_X, y, **fit_parameters)

        # Set post-processing parameters
        self._set_post_processing_params()

        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        # Convert the tree inference with Numpy operators
        self._tree_inference, self.output_quantizers, self.onnx_model_ = tree_to_numpy(
            self.sklearn_model,
            q_X[:1],
            framework=self.framework,
            output_n_bits=self.n_bits,
        )

        self._is_fitted = True

        return self

    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        q_X = numpy.zeros_like(X, dtype=numpy.int64)

        # Quantize using the learned quantization parameters for each feature
        for i, input_quantizer in enumerate(self.input_quantizers):
            q_X[:, i] = input_quantizer.quant(X[:, i])

        assert q_X.dtype == numpy.int64, "Inputs were not quantized to int64 values"
        return q_X

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        q_y_preds = self.output_quantizers[0].dequant(q_y_preds)
        return q_y_preds

    def _get_compiler(self) -> Compiler:
        assert self._tree_inference is not None, self._is_not_fitted_error_message()

        # Generate the proxy function to compile
        _tree_inference_proxy, parameters_mapping = generate_proxy_function(
            self._tree_inference, ["inputs"]
        )

        # Create the compiler instance
        compiler = Compiler(
            _tree_inference_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
        )

        return compiler

    def compile(self, *args, **kwargs) -> Circuit:
        BaseEstimator.compile(self, *args, **kwargs)

        # Check that the graph only has a single output
        # Ignore mypy issue as `self.fhe_circuit` is set in the super
        output_graph = self.fhe_circuit.graph.ordered_outputs()  # type: ignore[union-attr]
        assert_true(
            len(output_graph) == 1,
            "graph has too many outputs",
        )

        # Check that the output is an integer
        dtype_output = output_graph[0].output.dtype
        assert_true(
            isinstance(dtype_output, Integer),
            f"output is {dtype_output} but an Integer is expected.",
        )
        return self.fhe_circuit

    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        assert self._tree_inference is not None, self._is_not_fitted_error_message()

        return self._tree_inference(q_X)[0]

    def predict(
        self, X: numpy.ndarray, fhe: Union[FheMode, str] = FheMode.DISABLE
    ) -> numpy.ndarray:
        y_pred = BaseEstimator.predict(self, X, fhe=fhe)
        y_pred = self.post_processing(y_pred)
        return y_pred

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Sum all tree outputs
        # Remove the sum once we handle multi-precision circuits
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/451
        y_preds = numpy.sum(y_preds, axis=-1)

        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")
        return y_preds


class BaseTreeRegressorMixin(BaseTreeEstimatorMixin, sklearn.base.RegressorMixin, ABC):
    """Mixin class for tree-based regressors.

    This class is used to create a tree-based regressor class that inherits from
    sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method
    for regressors.
    """


class BaseTreeClassifierMixin(
    BaseClassifier, BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin, ABC
):
    """Mixin class for tree-based classifiers.

    This class is used to create a tree-based classifier class that inherits from
    sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method
    for classifiers.

    Additionally, this class adjusts some of the tree-based base class's methods in order to make
    them compliant with classification workflows.
    """


# pylint: disable=invalid-name,too-many-instance-attributes
class SklearnLinearModelMixin(BaseEstimator, sklearn.base.BaseEstimator, ABC):
    """A Mixin class for sklearn linear models with FHE.

    This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's
    `get_params` and `set_params` methods.
    """

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if getattr(klass, "_is_a_public_cml_model", False):
                _LINEAR_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, n_bits: Union[int, Dict[str, int]] = 8):
        """Initialize the FHE linear model.

        Args:
            n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
                for n_bits, the value will be used for quantizing inputs and weights. If a dict is
                passed, then it should contain "op_inputs" and "op_weights" as keys with
                corresponding number of quantization bits so that:
                    - op_inputs : number of bits to quantize the input values
                    - op_weights: number of bits to quantize the learned parameters
                Default to 8.
        """
        self.n_bits: Union[int, Dict[str, int]] = n_bits

        #: The equivalent fitted float model. Is None if the model is not fitted.
        self.sklearn_model: Optional[Any] = None

        #: The quantizer to use for quantizing the model's weights
        self._weight_quantizer: Optional[UniformQuantizer] = None

        #: The scale to use for dequantizing the predicted outputs
        self._output_scale: Optional[numpy.float64] = None

        #: The zero-point to use for dequantizing the predicted outputs
        self._output_zero_point: Optional[Union[numpy.ndarray, int]] = None

        #: The model's quantized weights
        self._q_weights: Optional[numpy.ndarray] = None

        #: The model's quantized bias
        self._q_bias: Optional[numpy.ndarray] = None

        BaseEstimator.__init__(self)

    def _sklearn_model_is_not_fitted_error_message(self) -> str:
        return (
            f"The underlying model (class: {self.sklearn_model_class}) is not fitted and thus "
            "cannot be converted quantized."
        )  # pragma: no cover

    def _set_onnx_model(self, test_input: numpy.ndarray):
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        self.onnx_model_ = hb_convert(
            self.sklearn_model,
            backend="onnx",
            test_input=test_input,
            extra_config={"onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT},
        ).model

        self._clean_graph()

    def _clean_graph(self):
        """Clean the ONNX graph from undesired nodes."""
        assert self.onnx_model_ is not None, self._is_not_fitted_error_message()

        # Remove cast operators as they are not needed
        remove_node_types(onnx_model=self.onnx_model_, op_types_to_remove=["Cast"])

    def _set_post_processing_params(self):
        self.post_processing_params = {
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
        }

    def fit(self, X, y, **fit_parameters) -> Any:
        # Reset for double fit
        self._is_fitted = False

        # LinearRegression handles multi-labels data
        X, y = check_X_y_and_assert_multi_output(X, y)

        # Fit the Scikit-Learn model
        self._fit_float_estimator(X, y, **fit_parameters)

        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        # Retrieve the ONNX graph
        self._set_onnx_model(X)

        # Convert the n_bits attribute into a proper dictionary
        n_bits = get_n_bits_dict(self.n_bits)

        input_n_bits = n_bits["op_inputs"]
        input_options = QuantizationOptions(n_bits=input_n_bits, is_signed=True)

        # Quantize the inputs and store the associated quantizer
        q_inputs = QuantizedArray(n_bits=input_n_bits, values=X, options=input_options)
        input_quantizer = q_inputs.quantizer
        self.input_quantizers.append(input_quantizer)

        weights_n_bits = n_bits["op_weights"]
        weight_options = QuantizationOptions(n_bits=weights_n_bits, is_signed=True)

        # Quantize the weights and store the associated quantizer
        # Transpose and expand are necessary in order to make sure the weight array has the correct
        # shape when calling the Gemm operator on it
        weights = self.sklearn_model.coef_.T
        q_weights = QuantizedArray(
            n_bits=n_bits["op_weights"],
            values=numpy.expand_dims(weights, axis=1) if len(weights.shape) == 1 else weights,
            options=weight_options,
        )
        self._q_weights = q_weights.qvalues
        weight_quantizer = q_weights.quantizer
        self._weight_quantizer = weight_quantizer

        # mypy
        assert input_quantizer.scale is not None
        assert weight_quantizer.scale is not None

        # Retrieve the scale and zero-point of the matmul's outputs, following the same steps from
        # the QuantizedGemm operator, which are based on equations detailed in
        # https://arxiv.org/abs/1712.05877
        self._output_scale = input_quantizer.scale * weight_quantizer.scale
        self._output_zero_point = input_quantizer.zero_point * (
            numpy.sum(self._q_weights, axis=0, keepdims=True)
            - X.shape[1] * weight_quantizer.zero_point
        )

        # Updating post-processing parameters
        self._set_post_processing_params()

        # Quantize the bias using the matmul's scale and zero-point, such that
        # q_bias = round((1/S)*bias + Z)
        # Contrary to the QuantizedGemm operator which handles the bias term as a floating point
        # (and thus fusing it with following TLUs), we need to quantize the bias term so that it
        # matches the same range of values as the matmul's outputs.
        self._q_bias = numpy.round(
            self.sklearn_model.intercept_ / self._output_scale + self._output_zero_point
        ).astype(numpy.int64)

        self._is_fitted = True

        return self

    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        q_X = self.input_quantizers[0].quant(X)

        assert q_X.dtype == numpy.int64, "Inputs were not quantized to int64 values"
        return q_X

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        # Retrieve the post-processing parameters
        output_scale = self.post_processing_params["output_scale"]
        output_zero_point = self.post_processing_params["output_zero_point"]

        # Dequantize the output.
        # Since the matmul and the bias both use the same scale and zero-points, we obtain that
        # y = S*(q_y - 2*Z)
        y_preds = output_scale * (
            q_y_preds - 2 * numpy.asarray(output_zero_point, dtype=numpy.float64)
        )

        return y_preds

    def _get_compiler(self) -> Compiler:
        # Define the inference function to compile.
        # This function can neither be a class method nor a static one because self we want to avoid
        # having self as a parameter while still being able to access some of its attribute
        def inference_to_compile(q_X: numpy.ndarray) -> numpy.ndarray:
            """Compile the circuit in FHE using only the inputs as parameters.

            Args:
                q_X (numpy.ndarray): The quantized input data

            Returns:
                numpy.ndarray: The circuit's outputs.
            """
            return self._inference(q_X)

        # Create the compiler instance
        compiler = Compiler(inference_to_compile, {"q_X": "encrypted"})

        return compiler

    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        # Quantizing weights and inputs makes an additional term appear in the inference function
        y_pred = q_X @ self._q_weights - self._weight_quantizer.zero_point * numpy.sum(
            q_X, axis=1, keepdims=True
        )
        y_pred += self._q_bias
        return y_pred


class SklearnLinearRegressorMixin(SklearnLinearModelMixin, sklearn.base.RegressorMixin, ABC):
    """A Mixin class for sklearn linear regressors with FHE.

    This class is used to create a linear regressor class that inherits from
    sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method
    for regressors.
    """


class SklearnLinearClassifierMixin(
    BaseClassifier, SklearnLinearModelMixin, sklearn.base.ClassifierMixin, ABC
):
    """A Mixin class for sklearn linear classifiers with FHE.

    This class is used to create a linear classifier class that inherits from
    sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method
    for classifiers.

    Additionally, this class adjusts some of the tree-based base class's methods in order to make
    them compliant with classification workflows.
    """

    def _clean_graph(self):
        assert self.onnx_model_ is not None, self._is_not_fitted_error_message()

        # Remove any operators following gemm, as they will be done in the clear
        clean_graph_after_node_op_type(self.onnx_model_, node_op_type="Gemm")
        SklearnLinearModelMixin._clean_graph(self)

    def decision_function(self, X, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        """Predict confidence scores.

        Args:
            X: The input values to predict.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete-ML python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            numpy.ndarray: The predicted confidence scores.
        """
        # Here, we want to use SklearnLinearModelMixin's `predict` method as confidence scores are
        # the dot product's output values, without any post-processing (as done in baseClassifier's
        # one)
        y_preds = SklearnLinearModelMixin.predict(self, X, fhe=fhe)
        return y_preds

    def predict_proba(self, X, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        y_preds = self.decision_function(X, fhe=fhe)
        y_preds = self.post_processing(y_preds)
        return y_preds
