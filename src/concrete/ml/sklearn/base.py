"""Base classes for all estimators."""

from __future__ import annotations

import copy
import os
import tempfile

# Disable pylint as some names like X and q_X are used, following scikit-Learn's standard. The file
# is also more than 1000 lines long.
# pylint: disable=too-many-lines,invalid-name
import warnings
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TextIO, Type, Union

import brevitas.nn as qnn

# pylint: disable-next=ungrouped-imports
import concrete.fhe as cp
import numpy
import onnx
import sklearn
import skorch.net
import torch
from brevitas.export.onnx.qonnx.manager import QONNXManager as BrevitasONNXManager
from concrete.fhe.compilation.artifacts import DebugArtifacts
from concrete.fhe.compilation.circuit import Circuit
from concrete.fhe.compilation.compiler import Compiler
from concrete.fhe.compilation.configuration import Configuration
from concrete.fhe.dtypes.integer import Integer
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_is_fitted
from xgboost.sklearn import XGBModel

from ..common.check_inputs import check_array_and_assert, check_X_y_and_assert_multi_output
from ..common.debugging.custom_assert import assert_true
from ..common.serialization.dumpers import dump, dumps
from ..common.utils import (
    USE_OLD_VL,
    FheMode,
    check_compilation_device_is_valid_and_is_cuda,
    check_execution_device_is_valid_and_is_cuda,
    check_there_is_no_p_error_options_in_configuration,
    generate_proxy_function,
    manage_parameters_for_pbs_errors,
)
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from ..onnx.onnx_model_manipulations import clean_graph_after_node_op_type, remove_node_types

# The sigmoid and softmax functions are already defined in the ONNX module and thus are imported
# here in order to avoid duplicating them.
from ..onnx.ops_impl import numpy_sigmoid, numpy_softmax
from ..quantization import (
    PostTrainingQATImporter,
    QuantizedArray,
    _get_n_bits_dict_trees,
    _inspect_tree_n_bits,
    get_n_bits_dict,
)
from ..quantization.quantized_module import QuantizedModule, _get_inputset_generator
from ..quantization.quantizers import (
    QuantizationOptions,
    UniformQuantizationParameters,
    UniformQuantizer,
)
from ..torch import NumpyModule
from .qnn_module import SparseQuantNeuralNetwork
from .tree_to_numpy import (
    get_equivalent_numpy_forward_from_onnx_tree,
    get_onnx_model,
    is_regressor_or_partial_regressor,
    onnx_fp32_model_to_quantized_model,
    tree_to_numpy,
)

# Disable pylint to import Hummingbird while ignoring the warnings
# pylint: disable=wrong-import-position,wrong-import-order
# Silence Hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402
from hummingbird.ml.operator_converters import constants  # noqa: E402

_ALL_SKLEARN_MODELS: Set[Type] = set()
_LINEAR_MODELS: Set[Type] = set()
_TREE_MODELS: Set[Type] = set()
_NEURALNET_MODELS: Set[Type] = set()
_NEIGHBORS_MODELS: Set[Type] = set()

# Define the supported types for both the input data and the target values. Since the Pandas
# library is currently only a dev dependencies, we cannot import it. We therefore need to use type
# strings and the `name-defined` mypy error to do so.
Data = Union[
    numpy.ndarray,
    torch.Tensor,
    "pandas.DataFrame",  # type: ignore[name-defined] # noqa: F821
    List,
]
Target = Union[
    numpy.ndarray,
    torch.Tensor,
    "pandas.DataFrame",  # type: ignore[name-defined] # noqa: F821
    "pandas.Series",  # type: ignore[name-defined] # noqa: F821
    List,
]

# Define QNN's attribute that will be auto-generated when fitting
QNN_AUTO_KWARGS = ["module__n_outputs", "module__input_dim"]

# Enable rounding feature for all tree-based models by default
# Note: This setting is fixed and cannot be altered by users
# However, for internal testing purposes, we retain the capability to disable this feature
os.environ["TREES_USE_ROUNDING"] = os.environ.get("TREES_USE_ROUNDING", "1")

# pylint: disable=too-many-public-methods


class BaseEstimator:
    """Base class for all estimators in Concrete ML.

    This class does not inherit from sklearn.base.BaseEstimator as it creates some conflicts
    with skorch in QuantizedTorchEstimatorMixin's subclasses (more specifically, the `get_params`
    method is not properly inherited).

    Attributes:
        _is_a_public_cml_model (bool): Private attribute indicating if the class is a public model
            (as opposed to base or mixin classes).
    """

    #: Base float estimator class to consider for the model. Is set for each subclasses.
    sklearn_model_class: Type

    _is_a_public_cml_model: bool = False

    def __init__(self):
        """Initialize the base class with common attributes used in all estimators.

        An underscore "_" is appended to attributes that were created while fitting the model. This
        is done in order to follow scikit-Learn's standard format. More information available
        in their documentation:
        https://scikit-learn.org/stable/developers/develop.html#:~:text=Estimated%20Attributes%C2%B6
        """
        #: The equivalent fitted float model. Is None if the model is not fitted.
        self.sklearn_model: Optional[sklearn.base.BaseEstimator] = None

        #: The list of quantizers, which contain all information necessary for applying uniform
        #: quantization to inputs and provide quantization/de-quantization functionalities. Is empty
        #: if the model is not fitted
        self.input_quantizers: List[UniformQuantizer] = []

        #: The list of quantizers, which contain all information necessary for applying uniform
        #: quantization to outputs and provide quantization/de-quantization functionalities. Is
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

        self._compiled_for_cuda: bool = False

        self.fhe_circuit_: Optional[Circuit] = None
        self.onnx_model_: Optional[onnx.ModelProto] = None

    def __getattr__(self, attr: str):
        """Get the model's attribute.

        If the attribute's name ends with an underscore ("_"), get the attribute from the underlying
        scikit-learn model as it represents a training attribute:
        https://scikit-learn.org/stable/glossary.html#term-attributes

        This method is only called if the attribute has not been found in the class instance:
        https://docs.python.org/3/reference/datamodel.html?highlight=getattr#object.__getattr__

        Args:
            attr (str): The attribute's name.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If the attribute cannot be found or is not a training attribute.
        """

        # If the attribute ends with a single underscore and can be found in the underlying
        # scikit-learn model (once fitted), retrieve its value
        # Enable non-training attributes as well once Concrete ML models initialize their
        # underlying scikit-learn models during initialization
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        if (
            attr.endswith("_")
            and not attr.endswith("__")
            and getattr(self, "sklearn_model", None) is not None
        ):
            return getattr(self.sklearn_model, attr)

        raise AttributeError(
            f"Attribute {attr} cannot be found in the Concrete ML {self.__class__.__name__} object "
            f"and is not a training attribute from the underlying scikit-learn "
            f"{self.sklearn_model_class} one. If the attribute is meant to represent one from that "
            f"latter, please check that the model is properly fitted."
        )

    # We need to specifically call the default __setattr__ method as QNN models still inherit from
    # skorch, which provides its own __setattr__ implementation and creates a cyclic loop
    # with __getattr__. Removing this inheritance once and for all should fix the issue
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
    def __setattr__(self, name: str, value: Any):
        """Set the value as a model attribute.

        Args:
            name (str): The attribute's name to consider.
            value (Any): The attribute's value to consider.
        """
        object.__setattr__(self, name, value)

    @abstractmethod
    def dump_dict(self) -> Dict[str, Any]:
        """Dump the object as a dict.

        Returns:
            Dict[str, Any]: Dict of serialized objects.
        """

    @classmethod
    @abstractmethod
    def load_dict(cls, metadata: Dict[str, Any]) -> BaseEstimator:
        """Load itself from a dict.

        Args:
            metadata (Dict[str, Any]): Dict of serialized objects.

        Returns:
            BaseEstimator: The loaded object.
        """

    def dumps(self) -> str:
        """Dump itself to a string.

        Returns:
            metadata (str): String of the serialized object.
        """
        return dumps(self)

    def dump(self, file: TextIO) -> None:
        """Dump itself to a file.

        Args:
            file (TextIO): The file to dump the serialized object into.
        """
        dump(self, file)

    @property
    def onnx_model(self) -> Optional[onnx.ModelProto]:
        """Get the ONNX model.

        Is None if the model is not fitted.

        Returns:
            onnx.ModelProto: The ONNX model.
        """
        assert isinstance(self.onnx_model_, onnx.ModelProto) or self.onnx_model_ is None
        return self.onnx_model_

    @property
    def fhe_circuit(self) -> Optional[Circuit]:
        """Get the FHE circuit.

        The FHE circuit combines computational graph, mlir, client and server into a single object.
        More information available in Concrete documentation
        (https://docs.zama.ai/concrete/get-started/terminology)
        Is None if the model is not fitted.

        Returns:
            Circuit: The FHE circuit.
        """
        assert isinstance(self.fhe_circuit_, Circuit) or self.fhe_circuit_ is None
        return self.fhe_circuit_

    def _sklearn_model_is_not_fitted_error_message(self) -> str:
        return (
            f"The underlying model (class: {self.sklearn_model_class}) is not fitted and thus "
            "cannot be quantized."
        )  # pragma: no cover

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

    def check_model_is_fitted(self) -> None:
        """Check if the model is fitted.

        Raises:
            AttributeError: If the model is not fitted.
        """
        if not self.is_fitted:
            raise AttributeError(self._is_not_fitted_error_message())

    @property
    def is_compiled(self) -> bool:
        """Indicate if the model is compiled.

        Returns:
            bool: If the model is compiled.
        """
        return self._is_compiled

    def _is_not_compiled_error_message(self) -> str:
        return (
            f"The {self.__class__.__name__} model is not compiled. "
            "Please run compile(...) first before executing the prediction in FHE."
        )

    def check_model_is_compiled(self) -> None:
        """Check if the model is compiled.

        Raises:
            AttributeError: If the model is not compiled.
        """
        if not self.is_compiled:
            raise AttributeError(self._is_not_compiled_error_message())

    def get_sklearn_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        This method is used to instantiate a scikit-learn model using the Concrete ML model's
        parameters. It does not override scikit-learn's existing `get_params` method in order to
        not break its implementation of `set_params`.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained
                subobjects that are estimators. Default to True.

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        # Here, the `get_params` method is the `BaseEstimator.get_params` method from scikit-learn,
        # which will become available once a subclass inherits from it. We therefore disable both
        # pylint and mypy as this behavior is expected
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        # pylint: disable-next=no-member
        params = super().get_params(deep=deep)  # type: ignore[misc]

        # Remove the n_bits parameters as this attribute is added by Concrete ML
        params.pop("n_bits", None)

        return params

    def _set_post_processing_params(self) -> None:
        """Set parameters used in post-processing."""
        self.post_processing_params = {}

    def _fit_sklearn_model(self, X: Data, y: Target, **fit_parameters):
        """Fit the model's scikit-learn equivalent estimator.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            **fit_parameters: Keyword arguments to pass to the scikit-learn estimator's fit method.

        Returns:
            The fitted scikit-learn estimator.
        """

        # Initialize the underlying scikit-learn model if it has not already been done or if
        # `warm_start` is set to False (for neural networks)
        # This model should be directly initialized in the model's __init__ method instead
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        if self.sklearn_model is None or not getattr(self, "warm_start", False):
            # Retrieve the init parameters
            params = self.get_sklearn_params()

            self.sklearn_model = self.sklearn_model_class(**params)

        # Fit the scikit-learn model
        self.sklearn_model.fit(X, y, **fit_parameters)

        return self.sklearn_model

    @abstractmethod
    def fit(self, X: Data, y: Target, **fit_parameters):
        """Fit the estimator.

        This method trains a scikit-learn estimator, computes its ONNX graph and defines the
        quantization parameters needed for proper FHE inference.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            **fit_parameters: Keyword arguments to pass to the float estimator's fit method.

        Returns:
            The fitted estimator.
        """

    # Several attributes and methods are called in `fit_benchmark` but will only be accessible
    # in subclasses, we therefore need to disable pylint and mypy from checking these no-member
    # issues
    # pylint: disable=no-member
    def fit_benchmark(
        self,
        X: Data,
        y: Target,
        random_state: Optional[int] = None,
        **fit_parameters,
    ):
        """Fit both the Concrete ML and its equivalent float estimators.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            random_state (Optional[int]): The random state to use when fitting. Defaults to None.
            **fit_parameters: Keyword arguments to pass to the float estimator's fit method.

        Returns:
            The Concrete ML and float equivalent fitted estimators.
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

        # Initialize the scikit-learn model
        sklearn_model = self.sklearn_model_class(**params)

        # Train the scikit-learn model
        sklearn_model.fit(X, y, **fit_parameters)

        # Update the Concrete ML model's parameters
        # Disable mypy attribute definition errors as this attribute is expected to be
        # initialized once the model inherits from skorch
        self.set_params(n_bits=self.n_bits, **params)  # type: ignore[attr-defined]

        # Train the Concrete ML model
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
        """De-quantize the output.

        This step ensures that the fit method has been called.

        Args:
            q_y_preds (numpy.ndarray): The quantized output values to de-quantize.

        Returns:
            numpy.ndarray: The de-quantized output values.
        """

    @abstractmethod
    def _get_module_to_compile(self) -> Union[Compiler, QuantizedModule]:
        """Retrieve the module instance to compile.

        Returns:
            Union[Compiler, QuantizedModule]: The module instance to compile.
        """

    def compile(
        self,
        X: Data,
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose: bool = False,
        device: str = "cpu",
    ) -> Circuit:
        """Compile the model.

        Args:
            X (Data): A representative set of input values used for building cryptographic
                parameters, as a Numpy array, Torch tensor, Pandas DataFrame or List. This is
                usually the training data-set or a sub-set of it.
            configuration (Optional[Configuration]): Options to use for compilation. Default
                to None.
            artifacts (Optional[DebugArtifacts]): Artifacts information about the compilation
                process to store for debugging. Default to None.
            show_mlir (bool): Indicate if the MLIR graph should be printed during compilation.
                Default to False.
            p_error (Optional[float]): Probability of error of a single PBS. A p_error value cannot
                be given if a global_p_error value is already set. Default to None, which sets this
                error to a default value.
            global_p_error (Optional[float]): Probability of error of the full circuit. A
                global_p_error value cannot be given if a p_error value is already set. This feature
                is not supported during the FHE simulation mode, meaning the probability is
                currently set to 0. Default to None, which sets this error to a default value.
            verbose (bool): Indicate if compilation information should be printed
                during compilation. Default to False.
            device: FHE compilation device, can be either 'cpu' or 'cuda'.

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

        use_gpu = check_compilation_device_is_valid_and_is_cuda(device)

        # Quantize the inputs
        q_X = self.quantize_input(X)

        # Generate the compilation input-set with proper dimensions
        inputset = _get_inputset_generator(q_X)

        # Retrieve the compiler instance
        module_to_compile = self._get_module_to_compile()

        # Compiling using a QuantizedModule requires different steps and should not be done here
        assert isinstance(module_to_compile, Compiler), (
            "Wrong module to compile. Expected to be of type `Compiler` but got "
            f"{type(module_to_compile)}."
        )

        # Enable input ciphertext compression
        enable_input_compression = os.environ.get("USE_INPUT_COMPRESSION", "1") == "1"
        # Enable evaluation key compression
        enable_key_compression = os.environ.get("USE_KEY_COMPRESSION", "1") == "1"

        self.fhe_circuit_ = module_to_compile.compile(
            inputset,
            configuration=configuration,
            artifacts=artifacts,
            show_mlir=show_mlir,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose,
            single_precision=False,
            use_gpu=use_gpu,
            compress_input_ciphertexts=enable_input_compression,
            compress_evaluation_keys=enable_key_compression,
        )

        self._is_compiled = True
        self._compiled_for_cuda = use_gpu

        # For mypy
        assert isinstance(self.fhe_circuit, Circuit)

        return self.fhe_circuit

    @abstractmethod
    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        """Inference function to consider when executing in the clear.

        Args:
            q_X (numpy.ndarray): The quantized input values.

        Returns:
            numpy.ndarray: The quantized predicted values.
        """

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        """Predict values for X, in FHE or in the clear.

        Args:
            X (Data): The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame
                or List.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete ML Python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            np.ndarray: The predicted values for X.
        """
        assert_true(
            FheMode.is_valid(fhe),
            "`fhe` mode is not supported. Expected one of 'disable' (resp. FheMode.DISABLE), "
            "'simulate' (resp. FheMode.SIMULATE) or 'execute' (resp. FheMode.EXECUTE). Got "
            f"{fhe}",
            ValueError,
        )

        # Check that the model is properly fitted
        self.check_model_is_fitted()

        check_execution_device_is_valid_and_is_cuda(self._compiled_for_cuda, fhe=fhe)

        # Ensure inputs are 2D
        if isinstance(X, (numpy.ndarray, torch.Tensor)) and X.ndim == 1:
            X = X.reshape((1, -1))

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

                # For mypy, even though we already check this with self.check_model_is_compiled()
                assert self.fhe_circuit is not None

                # If the inference should be executed using simulation
                if fhe == "simulate":
                    is_crt_encoding = self.fhe_circuit.statistics["packing_key_switch_count"] != 0

                    # If the virtual library method should be used
                    # For now, use the virtual library when simulating
                    # circuits that use CRT  encoding because the official simulation is too slow
                    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4391
                    if USE_OLD_VL or is_crt_encoding:
                        predict_method = partial(
                            self.fhe_circuit.graph, p_error=self.fhe_circuit.p_error
                        )  # pragma: no cover

                    # Else, use the official simulation method
                    else:
                        predict_method = self.fhe_circuit.simulate

                # Else, use the FHE execution method
                else:
                    predict_method = self.fhe_circuit.encrypt_run_decrypt

                # Execute the inference in FHE or with simulation
                q_y_pred_i = predict_method(q_X_i)
                assert isinstance(q_y_pred_i, numpy.ndarray)
                q_y_pred_list.append(q_y_pred_i[0])

            q_y_pred = numpy.array(q_y_pred_list)

        # Else, the prediction is simulated in the clear
        else:
            q_y_pred = self._inference(q_X)

        # De-quantize the predicted values in the clear
        y_pred = self.dequantize_output(q_y_pred)
        assert isinstance(y_pred, numpy.ndarray)
        return y_pred

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the de-quantized predictions.

        This post-processing step can include operations such as applying the sigmoid or softmax
        function for classifiers, or summing an ensemble's outputs. These steps are done in the
        clear because of current technical constraints. They most likely will be integrated in the
        FHE computations in the future.

        For some simple models such a linear regression, there is no post-processing step but the
        method is kept to make the API consistent for the client-server API. Other models might
        need to use attributes stored in `post_processing_params`.

        Args:
            y_preds (numpy.ndarray): The de-quantized predictions to post-process.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        assert isinstance(y_preds, numpy.ndarray), "Output predictions must be an array."

        return y_preds


# This class only is an equivalent of BaseEstimator applied to classifiers, therefore not all
# methods are implemented and we need to disable pylint from checking that
# pylint: disable-next=abstract-method
class BaseClassifier(BaseEstimator):
    """Base class for linear and tree-based classifiers in Concrete ML.

    This class inherits from BaseEstimator and modifies some of its methods in order to align them
    with classifier behaviors. This notably include applying a sigmoid/softmax post-processing to
    the predicted values as well as handling a mapping of classes in case they are not ordered.
    """

    # Remove in our next release major release
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3994
    @property
    def target_classes_(self) -> Optional[numpy.ndarray]:  # pragma: no cover
        """Get the model's classes.

        Using this attribute is deprecated.

        Returns:
            Optional[numpy.ndarray]: The model's classes.
        """
        warnings.warn(
            "Attribute 'target_classes_' is now deprecated and will be removed in a future "
            "version. Please use 'classes_' instead.",
            category=UserWarning,
            stacklevel=2,
        )

        return self.classes_

    # Remove in our next release major release
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3994
    @property
    def n_classes_(self) -> int:  # pragma: no cover
        """Get the model's number of classes.

        Using this attribute is deprecated.

        Returns:
            int: The model's number of classes.
        """

        # Tree-based classifiers from scikit-learn provide a `n_classes_` attribute
        if self.sklearn_model is not None and hasattr(self.sklearn_model, "n_classes_"):
            return self.sklearn_model.n_classes_

        warnings.warn(
            "Attribute 'n_classes_' is now deprecated and will be removed in a future version. "
            "Please use 'len(classes_)' instead.",
            category=UserWarning,
            stacklevel=2,
        )

        return len(self.classes_)

    def fit(self, X: Data, y: Target, **fit_parameters):
        X, y = check_X_y_and_assert_multi_output(X, y)

        # Retrieve the different target classes
        classes = numpy.unique(y)

        # Make sure y contains at least two classes
        assert_true(len(classes) > 1, "You must provide at least 2 classes in y.")

        # Change to composition in order to avoid diamond inheritance and indirect super() calls
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3249
        return super().fit(X, y, **fit_parameters)  # type: ignore[safe-super]

    def predict_proba(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        """Predict class probabilities.

        Args:
            X (Data): The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame
                or List.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete ML Python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return super().predict(X, fhe=fhe)

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Compute the predicted probabilities
        y_proba = self.predict_proba(X, fhe=fhe)

        # Retrieve the class with the highest probability
        y_preds = numpy.argmax(y_proba, axis=1)

        return self.classes_[y_preds]

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        y_preds = super().post_processing(y_preds)

        # If the prediction array is 1D, which happens with some models such as XGBCLassifier or
        # LogisticRegression models, we have a binary classification problem
        n_classes = y_preds.shape[1] if y_preds.ndim > 1 and y_preds.shape[1] > 1 else 2

        # For binary classification problem, apply the sigmoid operator
        if n_classes == 2:
            y_preds = numpy_sigmoid(y_preds)[0]

            # If the prediction array is 1D, transform the output into a 2D array [1-p, p],
            # with p the initial output probabilities
            # This is similar to what is done in scikit-learn
            if y_preds.ndim == 1 or y_preds.shape[1] == 1:
                y_preds = y_preds.flatten()
                return numpy.vstack([1 - y_preds, y_preds]).T

        # Else, apply the softmax operator
        else:
            y_preds = numpy_softmax(y_preds)[0]

        return y_preds


# Pylint complains that this method does not override the `dump_dict` and `load_dict` methods. This
# is expected as the QuantizedTorchEstimatorMixin class is not supposed to be used as such. This
# disable could probably be removed when refactoring the serialization of models
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3250
# pylint: disable-next=abstract-method,too-many-instance-attributes
class QuantizedTorchEstimatorMixin(BaseEstimator):
    """Mixin that provides quantization for a torch module and follows the Estimator API."""

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if getattr(klass, "_is_a_public_cml_model", False):
                _NEURALNET_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self):
        #: The quantized module used to store the quantized parameters. Is empty if the model is
        #: not fitted.
        self.quantized_module_ = QuantizedModule()

        #: The input dimension in the underlying model
        self.module__input_dim: Optional[int] = None

        #: The number of outputs in the underlying model
        self.module__n_outputs: Optional[int] = None

        BaseEstimator.__init__(self)

    @property
    def base_module(self) -> SparseQuantNeuralNetwork:
        """Get the Torch module.

        Returns:
            SparseQuantNeuralNetwork: The fitted underlying module.
        """
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        return self.sklearn_model.module_

    @property
    def input_quantizers(self) -> List[UniformQuantizer]:
        """Get the input quantizers.

        Returns:
            List[UniformQuantizer]: The input quantizers.
        """
        return self.quantized_module_.input_quantizers

    @input_quantizers.setter
    def input_quantizers(self, value: List[UniformQuantizer]) -> None:
        self.quantized_module_.input_quantizers = value

    @property
    def output_quantizers(self) -> List[UniformQuantizer]:
        """Get the output quantizers.

        Returns:
            List[UniformQuantizer]: The output quantizers.
        """
        return self.quantized_module_.output_quantizers

    @output_quantizers.setter
    def output_quantizers(self, value: List[UniformQuantizer]) -> None:
        self.quantized_module_.output_quantizers = value

    @property
    def fhe_circuit(self) -> Circuit:
        return self.quantized_module_.fhe_circuit

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        This method is overloaded in order to make sure that auto-computed parameters are not
        considered when cloning the model (e.g during a GridSearchCV call).

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        # Retrieve the skorch estimator's init parameters
        # Here, the `get_params` method is the `NeuralNet.get_params` method from skorch, which
        # will become available once a subclass inherits from it. We therefore disable both pylint
        # and mypy as this behavior is expected
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        # pylint: disable-next=no-member
        params = super().get_params(deep)  # type: ignore[misc]

        # Remove `module` since it is automatically set to SparseQuantNeuralNetImpl. Therefore,
        # we don't need to pass module again when cloning this estimator
        params.pop("module", None)

        # Remove the parameters that are auto-computed by `fit` as well
        for kwarg in QNN_AUTO_KWARGS:
            params.pop(kwarg, None)

        return params

    def get_sklearn_params(self, deep: bool = True) -> Dict:
        # Retrieve the skorch estimator's init parameters
        # Here, the `get_params` method is the `NeuralNet.get_params` method from skorch, which
        # will become available once a subclass inherits from it. We therefore disable both pylint
        # and mypy as this behavior is expected
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        # pylint: disable-next=no-member
        params = super().get_params(deep=deep)  # type: ignore[misc]

        # Set the quantized module to SparseQuantNeuralNetwork
        params["module"] = SparseQuantNeuralNetwork

        return params

    def _fit_sklearn_model(self, X: Data, y: Target, **fit_parameters):
        super()._fit_sklearn_model(X, y, **fit_parameters)

        # Make pruning permanent by removing weights associated to pruned neurons
        self.base_module.make_pruning_permanent()

    def fit(self, X: Data, y: Target, **fit_parameters):
        """Fit he estimator.

        If the module was already initialized, the module will be re-initialized unless
        `warm_start` is set to True. In addition to the torch training step, this method performs
        quantization of the trained Torch model using Quantization Aware Training (QAT).

        Values of dtype float64 are not supported and will be casted to float32.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            **fit_parameters: Keyword arguments to pass to skorch's fit method.

        Returns:
            The fitted estimator.
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
                # The parameter could be a list, numpy.ndarray or tensor
                if isinstance(attr_value, list):
                    attr_value = numpy.asarray(attr_value, numpy.float32)

                if isinstance(attr_value, numpy.ndarray):
                    setattr(self, arg_name, torch.from_numpy(attr_value).float())

                assert_true(
                    isinstance(getattr(self, arg_name), torch.Tensor),
                    f"Parameter `{arg_name}` must be a numpy.ndarray, list or torch.Tensor",
                )

        # Fit the model by using skorch's fit
        self._fit_sklearn_model(X, y, **fit_parameters)

        # Export the brevitas model to ONNX
        output_onnx_file_path = Path(tempfile.mkstemp(suffix=".onnx")[1])

        BrevitasONNXManager.export(
            self.base_module,
            input_shape=X[[0], ::].shape,
            export_path=str(output_onnx_file_path),
            keep_initializers_as_inputs=False,
            opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
        )

        self.onnx_model_ = onnx.load(str(output_onnx_file_path))

        output_onnx_file_path.unlink()

        # Create corresponding numpy model
        numpy_model = NumpyModule(self.onnx_model_, torch.tensor(X[[0], ::]))

        # Set the quantization bits for import
        # Note that the ONNXConverter will use a default value for network input bits
        # Furthermore, Brevitas ONNX contains bit-widths in the ONNX file
        # which override the bit-width that we pass here
        # Thus, this parameter is only used to check consistency during import (onnx file vs import)
        n_bits = self.base_module.n_a_bits

        # Import the quantization aware trained model
        qat_model = PostTrainingQATImporter(n_bits, numpy_model)

        self.quantized_module_ = qat_model.quantize_module(X)

        # Set post-processing params
        self._set_post_processing_params()

        self._is_fitted = True

        return self

    def _get_equivalent_float_module(self) -> torch.nn.Sequential:
        """Build a topologically equivalent Torch module that can be used on floating points.

        Returns:
            float_module (torch.nn.Sequential): The equivalent float module.
        """
        # Instantiate a new sequential module
        float_module = torch.nn.Sequential()

        layer_index = -1

        # Iterate over the model's sub-modules
        for module in self.base_module.features:

            # If the module is not a QuantIdentity, it is either a QuantLinear or an activation
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

                # Else, it is a module representing the activation function, which needs to be
                # added as well
                else:
                    activation_name = f"act{layer_index}"
                    float_module.add_module(activation_name, module)

        return float_module

    def _get_equivalent_float_estimator(self) -> skorch.net.NeuralNet:
        """Initialize a topologically equivalent estimator that can be used on floating points.

        Returns:
            float_estimator (skorch.net.NeuralNet): An instance of the equivalent float estimator.
        """
        # Retrieve the skorch estimator's init parameters
        sklearn_params = self.get_params()

        # Retrieve all parameters related to the module
        module_param_names = [name for name in sklearn_params if "module__" in name]

        # Remove all parameters related to the module
        for name in module_param_names:
            sklearn_params.pop(name, None)

        # Retrieve the estimator's float topological equivalent module
        float_module = self._get_equivalent_float_module()

        # Instantiate the float estimator
        float_estimator = self.sklearn_model_class(float_module, **sklearn_params)

        return float_estimator

    def fit_benchmark(
        self, X: Data, y: Target, random_state: Optional[int] = None, **fit_parameters
    ):
        """Fit the quantized estimator as well as its equivalent float estimator.

        This function returns both the quantized estimator (itself) as well as its non-quantized
        (float) equivalent, which are both trained separately. This method differs from the
        BaseEstimator's `fit_benchmark` method as QNNs use QAT instead of PTQ. Hence, here, the
        float model is topologically equivalent as we have less control over the influence of QAT
        over the weights.

        Values of dtype float64 are not supported and will be casted to float32.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data,  as a Numpy array, Torch tensor, Pandas DataFrame Pandas
                Series or List.
            random_state (Optional[int]): The random state to use when fitting. However, skorch
                does not handle such a parameter and setting it will have no effect. Defaults
                to None.
            **fit_parameters: Keyword arguments to pass to skorch's fit method.

        Returns:
            The Concrete ML and equivalent skorch fitted estimators.
        """

        assert (
            random_state is None
        ), "Neural Network models do not support random_state as a parameter when fitting."

        # Fit the quantized estimator
        self.fit(X, y, **fit_parameters)

        # Instantiate the equivalent float estimator
        float_estimator = self._get_equivalent_float_estimator()

        # Fit the float equivalent estimator
        float_estimator.fit(X, y, **fit_parameters)

        return self, float_estimator

    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        q_X = self.quantized_module_.quantize_input(X)

        assert numpy.array(q_X).dtype == numpy.int64, "Inputs were not quantized to int64 values"
        assert isinstance(q_X, numpy.ndarray)
        return q_X

    def dequantize_output(self, *q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        result = self.quantized_module_.dequantize_output(*q_y_preds)
        assert isinstance(result, numpy.ndarray)
        return result

    def _get_module_to_compile(self) -> Union[Compiler, QuantizedModule]:
        return self.quantized_module_

    def compile(
        self,
        X: Data,
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose: bool = False,
        device: str = "cpu",
    ) -> Circuit:
        # Reset for double compile
        self._is_compiled = False

        # Check that the model is correctly fitted
        self.check_model_is_fitted()

        # Cast pandas, list or torch to numpy
        X = check_array_and_assert(X)

        # Retrieve the module instance to compile
        module_to_compile = self._get_module_to_compile()

        # Compile the QuantizedModule
        module_to_compile.compile(
            X,
            configuration=configuration,
            artifacts=artifacts,
            show_mlir=show_mlir,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose,
            device=device,
        )

        # Make sure that no avoidable TLUs are found in the built-in model
        succ = list(self.fhe_circuit.graph.graph.successors(self.fhe_circuit.graph.input_nodes[0]))
        assert_true(
            not any(s.converted_to_table_lookup for s in succ),
            "The compiled circuit currently applies some lookup tables on input nodes but this "
            "should be avoided. Please check the underlying nn.Module.",
        )

        self._is_compiled = True

        return self.fhe_circuit

    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        result = self.quantized_module_.quantized_forward(q_X)
        assert isinstance(result, numpy.ndarray)
        return result

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Cast the predictions to float32 in order to match Torch's softmax outputs
        return y_preds.astype(numpy.float32)

    def prune(self, X: Data, y: Target, n_prune_neurons_percentage: float, **fit_params):
        """Prune a copy of this Neural Network model.

        This can be used when the number of neurons on the hidden layers is too high. For example,
        when creating a Neural Network model with `n_hidden_neurons_multiplier` high (3-4), it
        can be used to speed up the model inference in FHE. Many times, up to 50% of
        neurons can be pruned without losing accuracy, when using this function to fine-tune
        an already trained model with good accuracy. This method should be used
        once good accuracy is obtained.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data,  as a Numpy array, Torch tensor, Pandas DataFrame Pandas
                Series or List.
            n_prune_neurons_percentage (float): The percentage of neurons to remove. A value of
                0 (resp. 1.0) means no (resp. all) neurons will be removed.
            fit_params: Additional parameters to pass to the underlying nn.Module's forward method.

        Returns:
            A new pruned copy of the Neural Network model.

        Raises:
            ValueError: If the model has not been trained or has already been pruned.
        """
        self.check_model_is_fitted()

        if self.base_module.n_prune_neurons_percentage > 0.0:
            raise ValueError(
                "Cannot apply structured pruning optimization to an already pruned model"
            )

        if n_prune_neurons_percentage >= 1.0 or n_prune_neurons_percentage < 0:
            raise ValueError(
                f"Valid values for `n_prune_neurons_percentage` are in the [0..1) range, but "
                f"{n_prune_neurons_percentage} was given."
            )

        pruned_model = clone(self)

        # Copy the input/output dims, as they are kept constant. These
        # are usually computed by .fit, but here we want to manually instantiate the model
        # before .fit() in order to fine-tune the original model
        pruned_model.module__input_dim = self.module__input_dim
        pruned_model.module__n_outputs = self.module__n_outputs

        # Create .module_ if the .module is a class (not an instance)
        pruned_model.sklearn_model.initialize()

        # Deactivate the default pruning
        pruned_model.base_module.make_pruning_permanent()

        # Load the original model
        pruned_model.base_module.load_state_dict(self.base_module.state_dict())

        # Set the new pruning amount
        pruned_model.base_module.n_prune_neurons_percentage = n_prune_neurons_percentage

        assert len(pruned_model.base_module.pruned_layers) == 0

        # Enable pruning again, this time with structured pruning
        pruned_model.base_module.enable_pruning()

        # The .module_ was initialized manually, prevent .fit (for both skorch and Concrete ML)
        # from creating a new one
        # Setting both attributes could be avoided by initializing `sklearn_model` in __init__
        # # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        pruned_model.warm_start = True
        pruned_model.sklearn_model.warm_start = True

        # Now, fine-tune the original module with structured pruning
        pruned_model.fit(X, y, **fit_params)

        return pruned_model


class BaseTreeEstimatorMixin(BaseEstimator, sklearn.base.BaseEstimator, ABC):
    """Mixin class for tree-based estimators.

    This class inherits from sklearn.base.BaseEstimator in order to have access to scikit-learn's
    `get_params` and `set_params` methods.
    """

    #: Model's base framework used, either 'xgboost' or 'sklearn'. Is set for each subclasses.
    framework: str

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if getattr(klass, "_is_a_public_cml_model", False):
                _TREE_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, n_bits: Union[int, Dict[str, int]]):
        """Initialize the TreeBasedEstimatorMixin.

        Args:
            n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
                for n_bits, the value will be used for quantizing inputs and leaves. If a dict is
                passed, then it should contain "op_inputs" and "op_leaves" as keys with
                corresponding number of quantization bits so that:
                    - op_inputs (mandatory): number of bits to quantize the input values
                    - op_leaves (optional): number of bits to quantize the leaves
                Default to 6.
        """

        # Check if 'n_bits' is a valid value.
        _inspect_tree_n_bits(n_bits)

        self.n_bits: Union[int, Dict[str, int]] = n_bits

        #: The model's inference function. Is None if the model is not fitted.
        self._tree_inference: Optional[Callable] = None

        #: Wether to perform the sum of the output's tree ensembles in FHE or not.
        # By default, the decision of the tree ensembles is made in clear (not in FHE).
        # This attribute should not be modified by users.
        self._fhe_ensembling = False

        BaseEstimator.__init__(self)

    @classmethod
    def from_sklearn_model(
        cls,
        sklearn_model: sklearn.base.BaseEstimator,
        X: Optional[numpy.ndarray] = None,
        n_bits: int = 10,
    ):
        """Build a FHE-compliant model using a fitted scikit-learn model.

        Args:
            sklearn_model (sklearn.base.BaseEstimator): The fitted scikit-learn model to convert.
            X (Optional[Data]): A representative set of input values used for computing quantization
                parameters, as a Numpy array, Torch tensor, Pandas DataFrame or List. This is
                usually the training data-set or a sub-set of it.
            n_bits (int): Number of bits to quantize the model. If an int is passed
                for n_bits, the value will be used for quantizing inputs and weights. If a dict is
                passed, then it should contain "op_inputs" and "op_weights" as keys with
                corresponding number of quantization bits so that:
                - op_inputs : number of bits to quantize the input values
                - op_weights: number of bits to quantize the learned parameters
                Default to 8.

        Returns:
            The FHE-compliant fitted model.
        """
        # Check that sklearn_model is a proper fitted scikit-learn model
        check_is_fitted(sklearn_model)

        # Extract scikit-learn's initialization parameters
        init_params = sklearn_model.get_params()
        model = cls(n_bits=n_bits, **init_params)
        model._is_fitted = True

        # Update the underlying scikit-learn model with the given fitted one
        model.sklearn_model = copy.deepcopy(sklearn_model)

        # Get the onnx model, all operations needed to load it properly will be done on it.
        n_features = model.n_features_in_
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4545
        # Execute with 2 example for efficiency in large data scenarios to prevent slowdown
        # but also to work around the HB export issue.
        dummy_input = numpy.zeros((2, n_features))
        framework = "xgboost" if isinstance(sklearn_model, XGBModel) else "sklearn"
        onnx_model = get_onnx_model(
            model=sklearn_model,
            x=dummy_input,
            framework=framework,
        )

        # Tree values pre-processing
        # i.e., mainly predictions quantization
        # but also rounding the threshold such that they are now integers
        model._set_post_processing_params()

        # Get the expected number of ONNX outputs in the sklearn model.
        expected_number_of_outputs = 1 if is_regressor_or_partial_regressor(model) else 2

        (
            onnx_model,
            lsbs_to_remove_for_trees,
            input_quantizers,
            output_quantizers,
        ) = onnx_fp32_model_to_quantized_model(
            onnx_model,
            n_bits,
            framework,
            expected_number_of_outputs,
            n_features,
            X,
        )

        model.input_quantizers = input_quantizers
        model.output_quantizers = output_quantizers

        model._tree_inference, model.onnx_model_ = get_equivalent_numpy_forward_from_onnx_tree(
            onnx_model, lsbs_to_remove_for_trees=lsbs_to_remove_for_trees
        )

        return model

    def fit(self, X: Data, y: Target, **fit_parameters):
        # Reset for double fit
        self._is_fitted = False
        self.input_quantizers = []
        self.output_quantizers = []

        X, y = check_X_y_and_assert_multi_output(X, y)

        q_X = numpy.zeros_like(X)

        # Convert the n_bits attribute into a proper dictionary
        self.n_bits = _get_n_bits_dict_trees(self.n_bits)

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            input_quantizer = QuantizedArray(
                n_bits=self.n_bits["op_inputs"], values=X[:, i]
            ).quantizer
            self.input_quantizers.append(input_quantizer)
            q_X[:, i] = input_quantizer.quant(X[:, i])

        # Fit the scikit-learn model
        self._fit_sklearn_model(q_X, y, **fit_parameters)

        # Set post-processing parameters
        self._set_post_processing_params()

        # Check that the underlying sklearn model has been set and fit
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        # Enable rounding feature
        enable_rounding = os.environ.get("TREES_USE_ROUNDING", "1") == "1"

        if not enable_rounding:
            warnings.simplefilter("always")
            warnings.warn(
                "Using Concrete tree-based models without the `rounding feature` is deprecated. "
                "Consider setting 'use_rounding' to `True` for making the FHE inference faster "
                "and key generation.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        # Convert the tree inference with Numpy operators
        self._tree_inference, self.output_quantizers, self.onnx_model_ = tree_to_numpy(
            self.sklearn_model,
            q_X,
            use_rounding=enable_rounding,
            fhe_ensembling=self._fhe_ensembling,
            framework=self.framework,
            output_n_bits=self.n_bits["op_leaves"],
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

        y_preds = self.output_quantizers[0].dequant(q_y_preds)
        assert isinstance(y_preds, numpy.ndarray)

        # If the preds have shape (n, 1), squeeze it to shape (n,) like in scikit-learn
        if y_preds.ndim == 2 and y_preds.shape[1] == 1:
            return y_preds.ravel()

        return y_preds

    def _get_module_to_compile(self) -> Union[Compiler, QuantizedModule]:
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
        assert self.fhe_circuit is not None
        return self.fhe_circuit

    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        assert self._tree_inference is not None, self._is_not_fitted_error_message()

        return self._tree_inference(q_X)[0]

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        y_pred = BaseEstimator.predict(self, X, fhe=fhe)
        y_pred = self.post_processing(y_pred)
        return y_pred

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Sum all tree outputs
        # Only apply the sum in the clear if it has not already been done in FHE
        if not self._fhe_ensembling:
            y_preds = numpy.sum(y_preds, axis=-1)

            assert isinstance(y_preds, numpy.ndarray), "Output predictions must be an array."

            # If the preds have shape (n, 1), squeeze it to shape (n,) like in scikit-learn
            if y_preds.ndim == 2 and y_preds.shape[1] == 1:
                return y_preds.ravel()

            return y_preds

        return super().post_processing(y_preds)

    def get_sklearn_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        This method is used to instantiate a scikit-learn model using the Concrete ML model's
        parameters. It does not override scikit-learn's existing `get_params` method in order to
        not break its implementation of `set_params`.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained
                subobjects that are estimators. Default to True.

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        # pylint: disable-next=no-member
        params = super().get_params(deep=deep)  # type: ignore[misc]

        params.pop("n_bits", None)
        if "1.1." in sklearn.__version__:
            params.pop("monotonic_cst", None)  # pragma: no cover

        return params


class BaseTreeRegressorMixin(BaseTreeEstimatorMixin, sklearn.base.RegressorMixin, ABC):
    """Mixin class for tree-based regressors.

    This class is used to create a tree-based regressor class that inherits from
    sklearn.base.RegressorMixin, which essentially gives access to scikit-learn's `score` method
    for regressors.
    """


class BaseTreeClassifierMixin(
    BaseClassifier, BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin, ABC
):
    """Mixin class for tree-based classifiers.

    This class is used to create a tree-based classifier class that inherits from
    sklearn.base.ClassifierMixin, which essentially gives access to scikit-learn's `score` method
    for classifiers.

    Additionally, this class adjusts some of the tree-based base class's methods in order to make
    them compliant with classification workflows.
    """


# pylint: disable=invalid-name,too-many-instance-attributes
class SklearnLinearModelMixin(BaseEstimator, sklearn.base.BaseEstimator, ABC):
    """A Mixin class for sklearn linear models with FHE.

    This class inherits from sklearn.base.BaseEstimator in order to have access to scikit-learn's
    `get_params` and `set_params` methods.
    """

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
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

        #: The quantizer to use for quantizing the model's weights
        self._weight_quantizer: Optional[UniformQuantizer] = None

        #: The model's quantized weights
        self._q_weights: Optional[numpy.ndarray] = None

        #: The model's quantized bias
        self._q_bias: Optional[numpy.ndarray] = None

        BaseEstimator.__init__(self)

    @classmethod
    def from_sklearn_model(
        cls,
        sklearn_model: sklearn.base.BaseEstimator,
        X: Data,
        n_bits: Union[int, Dict[str, int]] = 8,
    ):
        """Build a FHE-compliant model using a fitted scikit-learn model.

        Args:
            sklearn_model (sklearn.base.BaseEstimator): The fitted scikit-learn model to convert.
            X (Data): A representative set of input values used for computing quantization
                parameters, as a Numpy array, Torch tensor, Pandas DataFrame or List. This is
                usually the training data-set or a sub-set of it.
            n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
                for n_bits, the value will be used for quantizing inputs and weights. If a dict is
                passed, then it should contain "op_inputs" and "op_weights" as keys with
                corresponding number of quantization bits so that:
                - op_inputs : number of bits to quantize the input values
                - op_weights: number of bits to quantize the learned parameters
                Default to 8.

        Returns:
            The FHE-compliant fitted model.
        """
        # Check that sklearn_model is a proper fitted scikit-learn model
        check_is_fitted(sklearn_model)

        # Extract scikit-learn's initialization parameters
        init_params = sklearn_model.get_params()

        # Ensure compatibility for both sklearn 1.1 and >=1.4
        # This parameter was removed in 1.4. If this package is installed
        # with sklearn 1.1 which has it, then remove it when
        # instantiating the 1.4 API compatible Concrete ML model
        init_params.pop("normalize", None)

        # Instantiate the Concrete ML model and update initialization parameters
        # This update is necessary as we currently store scikit-learn attributes in Concrete ML
        # classes during initialization (for example: link or power attributes in GLMs)
        # Without it, these attributes will have default values instead of the ones used by the
        # scikit-learn models
        # This should be fixed once Concrete ML models initialize their underlying scikit-learn
        # models during initialization
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        model = cls(n_bits=n_bits, **init_params)

        # Update the underlying scikit-learn model with the given fitted one
        model.sklearn_model = copy.deepcopy(sklearn_model)

        # Compute the quantization parameters
        return model._quantize_model(X)

    def _set_onnx_model(self, test_input: numpy.ndarray) -> None:
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        # Check that the underlying sklearn model has been set and fit
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        self.onnx_model_ = hb_convert(
            self.sklearn_model,
            backend="onnx",
            test_input=test_input,
            extra_config={"onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT},
        ).model

        self._clean_graph()

    def _clean_graph(self) -> None:
        """Clean the ONNX graph from undesired nodes."""
        assert self.onnx_model_ is not None, self._is_not_fitted_error_message()

        # Remove cast operators as they are not needed
        remove_node_types(onnx_model=self.onnx_model_, op_types_to_remove=["Cast"])

    def fit(self, X: Data, y: Target, **fit_parameters):
        # Reset for double fit
        self._is_fitted = False
        self.input_quantizers = []
        self.output_quantizers = []

        # LinearRegression handles multi-labels data
        X, y = check_X_y_and_assert_multi_output(X, y)

        # Fit the scikit-learn model
        self._fit_sklearn_model(X, y, **fit_parameters)

        # Compute the quantization parameters
        return self._quantize_model(X)

    def _quantize_model(self, X):
        """Compute quantization parameters.

        Args:
            X (Data): A representative set of input values used for computing quantization
                parameters, as a Numpy array, Torch tensor, Pandas DataFrame or List. This is
                usually the training data-set or a sub-set of it.

        Returns:
            The FHE-compliant fitted model.
        """
        # Check that the underlying sklearn model has been set and fit
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
        self.input_quantizers = [input_quantizer]

        weights_n_bits = n_bits["op_weights"]
        weight_options = QuantizationOptions(n_bits=weights_n_bits, is_signed=True)

        # Quantize the weights and store the associated quantizer
        # Transpose and expand are necessary in order to make sure the weight array has the correct
        # shape when calling the Gemm operator on it
        weights = self.sklearn_model.coef_.T
        q_weights = QuantizedArray(
            n_bits=n_bits["op_weights"],
            values=(numpy.expand_dims(weights, axis=1) if len(weights.shape) == 1 else weights),
            options=weight_options,
        )
        self._q_weights = q_weights.qvalues
        weight_quantizer = q_weights.quantizer
        self._weight_quantizer = weight_quantizer

        # mypy
        assert input_quantizer.scale is not None
        assert weight_quantizer.scale is not None

        # Compute the scale and zero-point of the matmul's outputs, following the same steps from
        # the QuantizedGemm operator, which are based on equations detailed in
        # https://arxiv.org/abs/1712.05877
        output_quant_params = UniformQuantizationParameters(
            scale=input_quantizer.scale * weight_quantizer.scale,
            zero_point=input_quantizer.zero_point
            * (
                numpy.sum(self._q_weights, axis=0, keepdims=True)
                - X.shape[1] * weight_quantizer.zero_point
            ),
            offset=0,
        )

        output_quantizer = UniformQuantizer(params=output_quant_params, no_clipping=True)

        # Quantize the bias using the matmul's scale and zero-point, such that
        # q_bias = round((1/S)*bias + Z)
        self._q_bias = output_quantizer.quant(self.sklearn_model.intercept_)

        # Since the matmul and the bias both use the same scale and zero-points, we obtain that
        # y = S*(q_y - 2*Z) when de-quantizing the values. We therefore need to multiply the initial
        # output zero_point by 2
        assert output_quantizer.zero_point is not None
        output_quantizer.zero_point *= 2
        self.output_quantizers = [output_quantizer]

        # Updating post-processing parameters
        self._set_post_processing_params()

        self._is_fitted = True

        return self

    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        q_X = self.input_quantizers[0].quant(X)

        assert q_X.dtype == numpy.int64, "Inputs were not quantized to int64 values"
        return q_X

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        # De-quantize the output values
        y_preds = self.output_quantizers[0].dequant(q_y_preds)
        assert isinstance(y_preds, numpy.ndarray)

        # If the preds have shape (n, 1), squeeze it to shape (n,) like in scikit-learn
        if y_preds.ndim == 2 and y_preds.shape[1] == 1:
            return y_preds.ravel()

        return y_preds

    def _get_module_to_compile(self) -> Union[Compiler, QuantizedModule]:
        # Define the inference function to compile.
        # This function can neither be a class method nor a static one because self we want to avoid
        # having self as a parameter while still being able to access some of its attribute
        def inference_to_compile(q_X: numpy.ndarray) -> numpy.ndarray:
            """Compile the circuit in FHE using only the inputs as parameters.

            Args:
                q_X (numpy.ndarray): The quantized input data

            Returns:
                numpy.ndarray: The circuit is outputs.
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
    sklearn.base.RegressorMixin, which essentially gives access to scikit-learn's `score` method
    for regressors.
    """


class SklearnLinearClassifierMixin(
    BaseClassifier, SklearnLinearModelMixin, sklearn.base.ClassifierMixin, ABC
):
    """A Mixin class for sklearn linear classifiers with FHE.

    This class is used to create a linear classifier class that inherits from
    sklearn.base.ClassifierMixin, which essentially gives access to scikit-learn's `score` method
    for classifiers.

    Additionally, this class adjusts some of the tree-based base class's methods in order to make
    them compliant with classification workflows.
    """

    def _clean_graph(self) -> None:
        assert self.onnx_model_ is not None, self._is_not_fitted_error_message()

        # Remove any operators following gemm, as they will be done in the clear
        assert self.onnx_model_ is not None
        clean_graph_after_node_op_type(self.onnx_model_, node_op_type="Gemm")
        SklearnLinearModelMixin._clean_graph(self)

    def decision_function(
        self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE
    ) -> numpy.ndarray:
        """Predict confidence scores.

        Args:
            X (Data): The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame
                or List.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete ML Python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            numpy.ndarray: The predicted confidence scores.
        """
        # Here, we want to use SklearnLinearModelMixin's `predict` method as confidence scores are
        # the dot product's output values, without any post-processing
        y_scores = SklearnLinearModelMixin.predict(self, X, fhe=fhe)

        return y_scores

    def predict_proba(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        y_scores = self.decision_function(X, fhe=fhe)
        y_proba = self.post_processing(y_scores)
        return y_proba

    # In scikit-learn, the argmax is done on the scores directly, not the probabilities
    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        # Compute the predicted scores
        y_scores = self.decision_function(X, fhe=fhe)

        # Retrieve the class with the highest score
        # If there is a single dimension, only compare the scores to 0
        if y_scores.ndim == 1:
            y_preds = (y_scores > 0).astype(int)
        else:
            y_preds = numpy.argmax(y_scores, axis=1)

        return self.classes_[y_preds]


class SklearnSGDRegressorMixin(SklearnLinearRegressorMixin):
    """A Mixin class for sklearn SGD regressors with FHE.

    This class is used to create a SGD regressor class what can be exported
    to ONNX using Hummingbird.
    """

    # Remove once Hummingbird supports SGDRegressor
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4100
    def _set_onnx_model(self, test_input: numpy.ndarray) -> None:
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        # Check that the underlying sklearn model has been set and fit
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        model_for_onnx = LinearRegression()
        model_for_onnx.coef_ = self.sklearn_model.coef_
        model_for_onnx.intercept_ = self.sklearn_model.intercept_

        self.onnx_model_ = hb_convert(
            model_for_onnx,
            backend="onnx",
            test_input=test_input,
            extra_config={"onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT},
        ).model

        self._clean_graph()


class SklearnSGDClassifierMixin(SklearnLinearClassifierMixin):
    """A Mixin class for sklearn SGD classifiers with FHE.

    This class is used to create a SGD classifier class what can be exported
    to ONNX using Hummingbird.
    """

    # Remove once Hummingbird supports SGDClassifier
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4100
    def _set_onnx_model(self, test_input: numpy.ndarray) -> None:
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        # Check that the underlying sklearn model has been set and fit
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        model_for_onnx = LogisticRegression()
        model_for_onnx.coef_ = self.sklearn_model.coef_
        model_for_onnx.intercept_ = self.sklearn_model.intercept_

        assert_true(
            hasattr(self.sklearn_model, "classes_"),
            "The fit method should have been called on this model",
        )
        model_for_onnx.classes_ = getattr(self.sklearn_model, "classes_", None)

        self.onnx_model_ = hb_convert(
            model_for_onnx,
            backend="onnx",
            test_input=test_input,
            extra_config={"onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT},
        ).model

        self._clean_graph()


# pylint: disable-next=invalid-name,too-many-instance-attributes
class SklearnKNeighborsMixin(BaseEstimator, sklearn.base.BaseEstimator, ABC):
    """A Mixin class for sklearn KNeighbors models with FHE.

    This class inherits from sklearn.base.BaseEstimator in order to have access to scikit-learn's
    `get_params` and `set_params` methods.
    """

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if getattr(klass, "_is_a_public_cml_model", False):
                _NEIGHBORS_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, n_bits: int = 3):
        """Initialize the FHE knn model.

        Args:
            n_bits (int): Number of bits to quantize the model. The value will be used for
                quantizing inputs and X_fit. Default to 3.
        """
        self.n_bits: int = n_bits

        # _q_fit_X: In distance metric algorithms, `_q_fit_X` stores the training set to compute
        # the similarity or distance measures. There is no `weights` attribute because there isn't
        # a training phase
        self._q_fit_X: numpy.ndarray

        # _y: Labels of `_q_fit_X`
        self._y: numpy.ndarray

        # _q_fit_X_quantizer: The quantizer to use for quantizing the model's training set
        self._q_fit_X_quantizer: Optional[UniformQuantizer] = None

        BaseEstimator.__init__(self)

    def _set_onnx_model(self, test_input: numpy.ndarray) -> None:
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        # Check that the underlying sklearn model has been set and fit
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        self.onnx_model_ = hb_convert(
            self.sklearn_model,
            backend="onnx",
            test_input=test_input,
            extra_config={
                "onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT,
                # pylint: disable-next=protected-access, no-member
                constants.BATCH_SIZE: self.sklearn_model._fit_X.shape[0],
            },
        ).model

        self._clean_graph()

    def _clean_graph(self) -> None:
        """Clean the ONNX graph from undesired nodes."""
        assert self.onnx_model_ is not None, self._is_not_fitted_error_message()

        # Remove cast operators as they are not needed
        remove_node_types(onnx_model=self.onnx_model_, op_types_to_remove=["Cast"])

    def fit(self, X: Data, y: Target, **fit_parameters):
        # Reset for double fit
        self._is_fitted = False
        self.input_quantizers = []
        self.output_quantizers = []

        # KNeighbors handles multi-labels data
        X, y = check_X_y_and_assert_multi_output(X, y)

        # Fit the scikit-learn model
        self._fit_sklearn_model(X, y, **fit_parameters)

        # Check that the underlying sklearn model has been set and fit
        assert self.sklearn_model is not None, self._sklearn_model_is_not_fitted_error_message()

        # Retrieve the ONNX graph
        self._set_onnx_model(X)

        # Quantize the inputs and store the associated quantizer
        input_options = QuantizationOptions(n_bits=self.n_bits, is_signed=True)
        q_inputs = QuantizedArray(n_bits=self.n_bits, values=X, options=input_options)
        input_quantizer = q_inputs.quantizer
        self.input_quantizers.append(input_quantizer)

        # Quantize the _fit_X and store the associated quantizer
        # pylint: disable-next=protected-access
        _fit_X = self.sklearn_model._fit_X
        # We assume that the inputs have the same distribution as the _fit_X
        q_fit_X = QuantizedArray(
            n_bits=self.n_bits,
            values=(numpy.expand_dims(_fit_X, axis=1) if len(_fit_X.shape) == 1 else _fit_X),
            options=input_options,
        )
        self._q_fit_X = q_fit_X.qvalues
        self._q_fit_X_quantizer = q_fit_X.quantizer

        # mypy
        assert self._q_fit_X_quantizer.scale is not None

        self._y = numpy.array(y)

        # We assume that the query has the same distribution as the data in _X_fit.
        # therefore, they use the same scaling and zero point.
        # https://arxiv.org/abs/1712.05877

        self.output_quant_params = UniformQuantizationParameters(
            scale=self._q_fit_X_quantizer.scale,
            zero_point=self._q_fit_X_quantizer.zero_point,
            offset=0,
        )

        output_quantizer = UniformQuantizer(params=self.output_quant_params, no_clipping=True)

        assert output_quantizer.zero_point is not None
        self.output_quantizers.append(output_quantizer)

        # Updating post-processing parameters
        self._set_post_processing_params()

        self._is_fitted = True

        return self

    def quantize_input(self, X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        q_X = self.input_quantizers[0].quant(X)

        assert q_X.dtype == numpy.int64, "Inputs were not quantized to int64 values"
        return q_X

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        # We compute the sorted argmax in FHE, which are integers.
        # No need to de-quantize the output values

        # If the preds have shape (n, 1), squeeze it to shape (n,) like in scikit-learn
        if q_y_preds.ndim > 1 and q_y_preds.shape[1] == 1:
            return q_y_preds.ravel()

        return q_y_preds

    def _get_module_to_compile(self) -> Union[Compiler, QuantizedModule]:
        # Define the inference function to compile.
        # This function can neither be a class method nor a static one because self we want to avoid
        # having self as a parameter while still being able to access some of its attribute
        def inference_to_compile(q_X: numpy.ndarray) -> numpy.ndarray:
            """Compile the circuit in FHE using only the inputs as parameters.

            Args:
                q_X (numpy.ndarray): The quantized input data

            Returns:
                numpy.ndarray: The circuit is outputs.
            """
            return self._inference(q_X)

        # Create the compiler instance
        compiler = Compiler(inference_to_compile, {"q_X": "encrypted"})

        return compiler

    @staticmethod
    def majority_vote(nearest_classes: numpy.ndarray):
        """Determine the most common class among nearest neighborsfor each query.

        Args:
            nearest_classes (numpy.ndarray): The class labels of the nearest neighbors for a query

        Returns:
            numpy.ndarray: The majority-voted class label for the corresponding query.
        """
        class_counts = numpy.bincount(nearest_classes)
        majority_votes = numpy.argmax(class_counts)

        return majority_votes

    def _inference(self, q_X: numpy.ndarray) -> numpy.ndarray:
        """Inference function.

        Args:
            q_X (numpy.ndarray): The quantized input values.

        Returns:
            numpy.ndarray: The quantized predicted values.
        """
        assert self._q_fit_X_quantizer is not None, self._is_not_fitted_error_message()

        def pairwise_euclidean_distance(q_X):
            # 1. Pairwise euclidean distance
            # dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
            return (
                numpy.sum(q_X**2, axis=1, keepdims=True)
                - 2 * q_X @ self._q_fit_X.T
                + numpy.expand_dims(numpy.sum(self._q_fit_X**2, axis=1), 0)
            )

        def topk_sorting(x, labels):
            """Argsort in FHE.

            Time complexity: O(nlog²(k))

            Args:
                x (numpy.ndarray): The quantized input values
                labels (numpy.ndarray): The labels of the training data-set

            Returns:
                numpy.ndarray: The argsort.
            """

            def gather1d(x, indices):
                """Select elements from the input array `x` using the provided `indices`.

                Args:
                    x (numpy.ndarray): The encrypted input array
                    indices (numpy.ndarray): The desired indexes

                Returns:
                    numpy.ndarray: The selected encrypted indexes.
                """
                arr = []
                for i in indices:
                    arr.append(x[i])
                enc_arr = cp.array(arr)
                return enc_arr

            def scatter1d(x, v, indices):
                """Rearrange elements of `x` with values from `v` at the specified `indices`.

                Args:
                    x (numpy.ndarray): The encrypted input array in which items will be updated
                    v (numpy.ndarray): The array containing values to be inserted into `x`
                        at the specified `indices`.
                    indices (numpy.ndarray): The indices indicating where to insert the elements
                        from `v` into `x`.

                Returns:
                    numpy.ndarray: The updated encrypted `x`
                """
                for idx, i in enumerate(indices):
                    x[i] = v[idx]
                return x

            comparisons = numpy.zeros(x.shape)
            labels = labels + cp.zeros(labels.shape)

            n, k = x.size, self.n_neighbors
            # Determine the number of stages for a sequence of length n
            ln2n = int(numpy.ceil(numpy.log2(n)))

            # Stage loop
            for t in range(ln2n - 1, -1, -1):
                # p: Determines the range of indexes to be compared in each pass.
                p = 2**t
                # r: Offset that adjusts the range of indexes to be compared and sorted in each pass
                r = 0
                # d: Comparison distance in each pass
                d = p
                # Number of passes for each stage
                for bq in range(ln2n - 1, t - 1, -1):
                    with cp.tag(f"Stage_{t}_pass_{bq}"):
                        q = 2**bq
                        # Determine the range of indexes to be compared
                        range_i = numpy.array(
                            [i for i in range(0, n - d) if i & p == r and comparisons[i] < k]
                        )
                        if len(range_i) == 0:
                            # Edge case, for k=1
                            continue

                        # Select 2 bitonic sequences `a` and `b` of length `d`
                        # a = x[range_i]: first bitonic sequence
                        # a_i = idx[range_i]: Indexes of a_i elements in the original x
                        a = gather1d(x, range_i)
                        # a_i = gather1d(idx, range_i)
                        # b = x[range_i + d]: Second bitonic sequence
                        # b_i = idx[range_i + d]: Indexes of b_i elements in the original x
                        b = gather1d(x, range_i + d)
                        # b_i = gather1d(idx, range_i + d)

                        labels_a = gather1d(labels, range_i)  #
                        labels_b = gather1d(labels, range_i + d)  # idx[range_i + d]

                        with cp.tag("diff"):
                            # Select max(a, b)
                            diff = b - a

                        with cp.tag("max_value"):
                            max_x = a + numpy.maximum(0, diff)

                        with cp.tag("swap_max_value"):
                            # Swap if a > b
                            # x[range_i] = max_x(a, b): First bitonic sequence gets min(a, b)
                            x = scatter1d(x, a + b - max_x, range_i)
                            # x[range_i + d] = min(a, b): Second bitonic sequence gets max(a, b)
                            x = scatter1d(x, max_x, range_i + d)

                        # Update labels array according to the max value
                        with cp.tag("max_label"):
                            is_a_greater_than_b = diff > 0
                            max_labels = labels_a + (labels_b - labels_a) * is_a_greater_than_b

                        with cp.tag("swap_max_label"):
                            labels = scatter1d(labels, labels_a + labels_b - max_labels, range_i)
                            labels = scatter1d(labels, max_labels, range_i + d)

                        # Update
                        with cp.tag("update"):
                            comparisons[range_i + d] = comparisons[range_i + d] + 1
                            # Reduce the comparison distance by half
                            d = q - p
                            r = p

            return labels[0 : self.n_neighbors]

        # 1. Pairwise_euclidiean distance
        with cp.tag("Original distance"):
            distance_matrix = pairwise_euclidean_distance(q_X)

        # The square root in the Euclidean distance calculation is not applied to speed up FHE
        # computations.
        # Being a monotonic function, it does not affect the logic of the calculation, notably for
        # the argsort.

        topk_labels = topk_sorting(distance_matrix.flatten(), self._y)
        return numpy.expand_dims(topk_labels, axis=0)

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Provide the majority vote among the topk labels of each point.

        For KNN, the de-quantization step is not required. Because _inference returns the label of
        the k-nearest neighbors.

        Args:
            y_preds (numpy.ndarray): The topk nearest labels for each point.

        Returns:
            numpy.ndarray: The majority vote for each point.
        """
        return numpy.array([self.majority_vote(y.flatten()) for y in y_preds])

    def get_topk_labels(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:
        """Return the K-nearest labels of each point.

        Args:
            X (Data): The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame
                or List.
            fhe (Union[FheMode, str]): The mode to use for prediction.
                Can be FheMode.DISABLE for Concrete ML Python inference,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values.
                Default to FheMode.DISABLE.

        Returns:
            numpy.ndarray: The K-Nearest labels for each point.
        """

        X = check_array_and_assert(X)

        topk_labels = []
        for query in X:
            query = numpy.expand_dims(query, 0)
            topk_labels.append(BaseEstimator.predict(self, query, fhe=fhe)[0])

        return numpy.array(topk_labels)

    def predict(self, X: Data, fhe: Union[FheMode, str] = FheMode.DISABLE) -> numpy.ndarray:

        X = check_array_and_assert(X)

        topk_labels = self.get_topk_labels(X, fhe)

        y_preds = self.post_processing(topk_labels)

        return y_preds


class SklearnKNeighborsClassifierMixin(SklearnKNeighborsMixin, sklearn.base.ClassifierMixin, ABC):
    """A Mixin class for sklearn KNeighbors classifiers with FHE.

    This class is used to create a KNeighbors classifier class that inherits from
    SklearnKNeighborsMixin and sklearn.base.ClassifierMixin.
    By inheriting from sklearn.base.ClassifierMixin, it allows this class to be recognized
    as a classifier."
    """
