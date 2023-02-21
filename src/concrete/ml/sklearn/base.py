"""Module that contains base classes for our libraries estimators."""
import copy
import functools
import tempfile

# https://github.com/zama-ai/concrete-ml-internal/issues/942
# Refactoring base.py. This file is more than 1000 lines.
# We use names like X and q_X
# pylint: disable=too-many-lines,invalid-name
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

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

from concrete.ml.quantization.quantized_module import QuantizedModule, _get_inputset_generator
from concrete.ml.quantization.quantizers import QuantizationOptions, UniformQuantizer

from ..common.check_inputs import check_array_and_assert, check_X_y_and_assert
from ..common.debugging.custom_assert import assert_true
from ..common.utils import (
    check_there_is_no_p_error_options_in_configuration,
    generate_proxy_function,
    is_classifier_or_partial_classifier,
    is_regressor_or_partial_regressor,
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

    input_quantizers: List[UniformQuantizer]
    output_quantizers: List[UniformQuantizer]
    fhe_circuit_: Optional[Circuit]
    onnx_model_: Optional[onnx.ModelProto]
    post_processing_params: Dict[str, Any]
    _is_a_public_cml_model: bool = False

    def __init__(self):
        """Initialize the base class with common attributes used in all estimators.

        An underscore "_" is appended to attributes that were created while fitting the model. This
        is done in order to follow Scikit-Learn's standard format. More information available
        in their documentation:
        https://scikit-learn.org/stable/developers/develop.html#:~:text=Estimated%20Attributes%C2%B6
        """
        #: input_quantizers (List[UniformQuantizer]): List of quantizers, which contain all
        #: information necessary for applying uniform quantization to inputs and provide
        #: quantization/dequantization functionalities. Is empty if the model was not fitted
        self.input_quantizers = []

        #: output_quantizers (List[UniformQuantizer]): List of quantizers, which contain all
        #: information necessary for applying uniform quantization to outputs and provide
        #: quantization/dequantization functionalities. Is empty if the model was not fitted
        self.output_quantizers = []

        #: post_processing_params (Dict[str, Any]): Parameters needed for post-processing the
        #: outputs. Can be empty if no post-processing operations are needed for the
        #: associated model
        self.post_processing_params = {}

        self.fhe_circuit_ = None
        self.onnx_model_ = None

    @property
    def onnx_model(self) -> Optional[onnx.ModelProto]:
        """Get the ONNX model.

        Is None if the model was not fitted.

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
        Is None if the model was not fitted.

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
    @abstractmethod
    def _is_fitted(self) -> bool:
        """Indicate if the model is fitted.

        Returns:
            bool: If the model is fitted.
        """

    def check_model_is_fitted(self):
        """Check if the model is fitted.

        Raises:
            AttributeError: If the model is not fitted.
        """
        if not self._is_fitted:
            raise AttributeError(
                f"The {self.__class__.__name__} model is not fitted. "
                "Please run fit(...) on proper arguments first.",
            )

    def check_model_is_compiled(self):
        """Check if the model is compiled.

        Raises:
            AttributeError: If the model is not compiled.
        """
        if self.fhe_circuit is None:
            raise AttributeError(
                f"The {self.__class__.__name__} model is not compiled. "
                "Please run compile(...) first before executing the prediction in FHE."
            )

    def get_sklearn_params(self, deep=True):
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
        # Pylint doesn't properly handle classes mixed with classes from a different file
        # Here, the `get_params` method is the `BaseEstimator.get_params` method from Scikit-Learn
        # pylint: disable-next=no-member
        params = super().get_params(deep=deep)

        # Remove the n_bits parameters as this attribute is added by Concrete-ML
        params.pop("n_bits", None)

        return params

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


class QuantizedTorchEstimatorMixin(BaseEstimator):
    """Mixin that provides quantization for a torch module and follows the Estimator API."""

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
                _NEURALNET_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(
        self,
    ):
        self.quantized_module_ = QuantizedModule()
        super().__init__()

    @property
    @abstractmethod
    def base_estimator_type(self):
        """Get the sklearn estimator that should be trained by the child class."""

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

    @property
    def _is_fitted(self):
        return self.quantized_module_.input_quantizers and self.quantized_module_.output_quantizers

    def quantize_input(self, X: numpy.ndarray):
        self.check_model_is_fitted()
        return self.quantized_module_.quantize_input(X)

    def dequantize_output(self, q_y_preds: numpy.ndarray):
        self.check_model_is_fitted()
        return self.quantized_module_.dequantize_output(q_y_preds)

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the dequantized predictions.

        Args:
            y_preds (numpy.ndarray): The dequantized predictions to post-process.

        Raises:
            ValueError: If the post-processing function is unknown.

        Returns:
            numpy.ndarray: The post-processed dequantized predictions.
        """
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

    def compile(
        self,
        X: numpy.ndarray,
        configuration: Optional[Configuration] = None,
        compilation_artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose_compilation: bool = False,
    ) -> Circuit:
        """Compile the model.

        Args:
            X (numpy.ndarray): the dequantized dataset
            configuration (Optional[Configuration]): the options for
                compilation
            compilation_artifacts (Optional[DebugArtifacts]): artifacts object to fill
                during compilation
            show_mlir (bool): whether or not to show MLIR during the compilation
            use_virtual_lib (bool): whether to compile using the virtual library that allows higher
                bitwidths
            p_error (Optional[float]): probability of error of a single PBS
            global_p_error (Optional[float]): probability of error of the full circuit. Not
                simulated by the VL, i.e., taken as 0
            verbose_compilation (bool): whether to show compilation information

        Returns:
            Circuit: the compiled Circuit.
        """
        self.check_model_is_fitted()

        # Cast pandas, list or torch to numpy
        X = check_array_and_assert(X)

        # Quantize the compilation input set using the quantization parameters computed in .fit()
        quantized_numpy_inputset = self.quantize_input(X)

        # Don't let the user shoot in her foot, by having p_error or global_p_error set in both
        # configuration and in direct arguments
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Call the compilation backend to produce the FHE inference circuit
        circuit = self.quantized_module_.compile(
            quantized_numpy_inputset,
            configuration=configuration,
            compilation_artifacts=compilation_artifacts,
            show_mlir=show_mlir,
            use_virtual_lib=use_virtual_lib,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose_compilation=verbose_compilation,
        )

        succ = list(circuit.graph.graph.successors(circuit.graph.input_nodes[0]))
        assert_true(
            not any(s.converted_to_table_lookup for s in succ),
            "The compiled circuit"
            " applies lookup tables on input nodes, please check the underlying nn.Module.",
        )

        return circuit

    def fit(self, X, y, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True). In addition to the
        torch training step, this method performs quantization of the trained torch model.

        Args:
            X : training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): labels associated with training data
            **fit_params: additional parameters that can be used during training, these are passed
                to the torch training interface

        Returns:
            self: the trained quantized estimator
        """
        # Reset the quantized module since quantization is lost during refit
        # This will make the .infer() function call into the Torch nn.Module
        # Instead of the quantized module
        self.quantized_module_ = QuantizedModule()
        multi_output = isinstance(y[0], list) if isinstance(y, list) else len(y.shape) > 1
        X, y = check_X_y_and_assert(X, y, multi_output=multi_output)

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

        # Call skorch fit that will train the network
        # This will instantiate the model class if it's not already done.
        super().fit(X, y, **fit_params)

        # Export the brevitas model to ONNX
        output_onnx_file_path = Path(tempfile.mkstemp(suffix=".onnx")[1])

        BrevitasONNXManager.export(
            self.base_module_to_compile,
            input_shape=X[[0], ::].shape,
            export_path=str(output_onnx_file_path),
            keep_initializers_as_inputs=False,
            opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
        )

        onnx_model = onnx.load(output_onnx_file_path)

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
        n_bits = self.n_bits_quant

        # Import the quantization aware trained model
        qat_model = PostTrainingQATImporter(n_bits, numpy_model)

        self.quantized_module_ = qat_model.quantize_module(X)

        # Set post-processing params
        self._update_post_processing_params()
        return self

    def _update_post_processing_params(self):
        """Update the post-processing parameters."""
        params = self._get_predict_nonlinearity()  # type: ignore
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

    # Disable pylint here because we add an additional argument to .predict,
    # with respect to the base class .predict method.
    # pylint: disable=arguments-differ
    def predict(self, X, execute_in_fhe=False):
        """Predict on user provided data.

        Predicts using the quantized clear or FHE classifier

        Args:
            X : input data, a numpy array of raw values (non quantized)
            execute_in_fhe : whether to execute the inference in FHE or in the clear

        Returns:
            y_pred : numpy ndarray with predictions

        """
        # By default, just return the result of predict_proba
        # which for linear models with no non-linearity applied simply returns
        # the decision function value
        return self.predict_proba(X, execute_in_fhe=execute_in_fhe)

    def predict_proba(self, X, execute_in_fhe=False):
        """Predict on user provided data, returning probabilities.

        Predicts using the quantized clear or FHE classifier

        Args:
            X : input data, a numpy array of raw values (non quantized)
            execute_in_fhe : whether to execute the inference in FHE or in the clear

        Returns:
            y_pred : numpy ndarray with probabilities (if applicable)
        """
        if execute_in_fhe:
            self.check_model_is_fitted()
            self.check_model_is_compiled()

            # Run over each element of X individually and aggregate predictions in a vector
            if X.ndim == 1:
                X = X.reshape((1, -1))

            X = check_array_and_assert(X)
            q_y_pred = None
            for idx, x in enumerate(X):
                q_x = self.quantize_input(x).reshape(1, -1)
                q_pred = self.fhe_circuit.encrypt_run_decrypt(q_x)
                if q_y_pred is None:
                    assert_true(
                        isinstance(q_pred, numpy.ndarray),
                        f"bad type {q_pred}, expected np.ndarray",
                    )
                    # pylint is lost: Instance of 'tuple' has no 'size' member (no-member)
                    # because it doesn't understand the Union in encrypt_run_decrypt
                    # pylint: disable=no-member
                    q_y_pred = numpy.zeros((X.shape[0], q_pred.size), numpy.float32)
                    # pylint: enable=no-member
                q_y_pred[idx, :] = q_pred

            # Dequantize the outputs and apply post processing
            y_pred = self.dequantize_output(q_y_pred)
            y_pred = self.post_processing(y_pred)
            return y_pred

        # For prediction in the clear we call the super class which, in turn,
        # will end up calling .infer of this class
        return super().predict_proba(X).astype(numpy.float32)

    # pylint: enable=arguments-differ

    def fit_benchmark(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs) -> Tuple[Any, Any]:
        """Fit the quantized estimator as well as its equivalent float estimator.

        This function returns both the quantized estimator (itself) as well as its non-quantized
        (float) equivalent, which are both trained separately. This is useful in order
        to compare performances between quantized and fp32 versions.

        Args:
            X : The training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): The labels associated with the training data
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.

        Returns:
            self: The trained quantized estimator
            fp32_model: The trained float equivalent estimator
        """
        # Fit the quantized estimator
        self.fit(X, y, *args, **kwargs)

        # Retrieve the Skorch estimator's training parameters
        estimator_parameters = self.get_sklearn_params()

        # Retrieve the estimator's float equivalent module
        float_module = self._get_equivalent_float_module()

        # Instantiate the float estimator
        skorch_estimator_type = self.base_estimator_type
        float_estimator = skorch_estimator_type(float_module, **estimator_parameters)

        # Fit the float estimator
        float_estimator.fit(X, y, *args, **kwargs)

        return self, float_estimator

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


class BaseTreeEstimatorMixin(BaseEstimator, sklearn.base.BaseEstimator):
    """Mixin class for tree-based estimators.

    This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's
    `get_params` and `set_params` methods.
    """

    n_bits: int
    random_state: Optional[Union[numpy.random.RandomState, int]] = None  # pylint: disable=no-member
    sklearn_alg: Callable[..., sklearn.base.BaseEstimator]
    sklearn_model: sklearn.base.BaseEstimator
    _tensor_tree_predict: Callable
    framework: str

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
                _TREE_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, n_bits: int):
        """Initialize the TreeBasedEstimatorMixin.

        Args:
            n_bits (int): Number of bits used for quantization.
        """
        self.n_bits: int = n_bits
        super().__init__()

    @property
    def _is_fitted(self):
        return self.input_quantizers and self.output_quantizers

    def quantize_input(self, X: numpy.ndarray):
        self.check_model_is_fitted()

        qX = numpy.zeros_like(X)

        # Quantize using the learned quantization parameters for each feature
        for i, q_x_ in enumerate(self.input_quantizers):
            qX[:, i] = q_x_.quant(X[:, i])

        return qX.astype(numpy.int64)

    def dequantize_output(self, q_y_preds: numpy.ndarray):
        self.check_model_is_fitted()

        q_y_preds = self.output_quantizers[0].dequant(q_y_preds)
        return q_y_preds

    def _update_post_processing_params(self):
        """Update the post processing parameters."""
        self.post_processing_params = {}

    def fit(self, X, y: numpy.ndarray, **kwargs) -> Any:
        """Fit the tree-based estimator.

        Args:
            X : training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): The target data.
            **kwargs: args for super().fit

        Returns:
            Any: The fitted model.
        """
        multi_output = isinstance(y[0], list) if isinstance(y, list) else len(y.shape) > 1
        X, y = check_X_y_and_assert(X, y, multi_output=multi_output)

        qX = numpy.zeros_like(X)
        self.input_quantizers = []

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i]).quantizer
            self.input_quantizers.append(q_x_)
            qX[:, i] = q_x_.quant(X[:, i])

        # Retrieve the init parameters
        params = self.get_sklearn_params()

        # Initialize the sklearn model
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(qX, y, **kwargs)

        # Set post-processing parameters
        self._update_post_processing_params()

        # Convert the tree inference with Numpy operators
        self._tensor_tree_predict, self.output_quantizers, self.onnx_model_ = tree_to_numpy(
            self.sklearn_model,
            qX[:1],
            framework=self.framework,
            output_n_bits=self.n_bits,
        )

        return self

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
        params = self.get_sklearn_params()  # type: ignore

        # Make sure the random_state is set or both algorithms will diverge
        # due to randomness in the training.
        if random_state is not None:
            params["random_state"] = random_state
        elif self.random_state is not None:
            params["random_state"] = self.random_state
        else:
            params["random_state"] = numpy.random.randint(0, 2**15)

        # Train the sklearn model without X quantized
        sklearn_model = self.sklearn_alg(**params)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        self.set_params(n_bits=self.n_bits, **params)
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model

    def _execute_in_fhe(self, qX: numpy.ndarray) -> numpy.ndarray:
        """Execute the FHE inference on the input data.

        Args:
            qX (numpy.ndarray): the input data quantized

        Returns:
            numpy.ndarray: the prediction as ordinals
        """
        self.check_model_is_compiled()

        y_preds = []
        for qX_i in qX:
            # Expected encrypt_run_decrypt output shape is (1, n_features) but qX_i is (n_features,)
            qX_i = numpy.expand_dims(qX_i, 0)

            # Ignore mypy issue as `check_model_is_compiled` makes sure that fhe_circuit is not None
            fhe_pred = self.fhe_circuit.encrypt_run_decrypt(qX_i)  # type: ignore[union-attr]
            y_preds.append(fhe_pred)

        # Concatenate on the n_examples dimension where shape is (n_examples, n_classes, n_trees)
        y_preds_array = numpy.concatenate(y_preds, axis=0)
        return y_preds_array

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict values for X.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE or not. Defaults to False.

        Returns:
            numpy.ndarray: The predicted values.
        """
        self.check_model_is_fitted()

        X = check_array_and_assert(X)

        # Quantize the input
        qX = self.quantize_input(X)

        # If the inference should be executed in FHE
        if execute_in_fhe:
            q_y_pred = self._execute_in_fhe(qX)

        # Else, the prediction is simulated in the clear
        else:
            q_y_pred = self._tensor_tree_predict(qX)[0]

        # Dequantize the predicted values and apply some post-processing in the clear
        y_pred = self.dequantize_output(q_y_pred)
        y_pred = self.post_processing(y_pred)

        return y_pred

    def compile(
        self,
        X: numpy.ndarray,
        configuration: Optional[Configuration] = None,
        compilation_artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose_compilation: bool = False,
    ) -> Circuit:
        """Compile the model.

        Args:
            X (numpy.ndarray): the dequantized dataset
            configuration (Optional[Configuration]): the options for
                compilation
            compilation_artifacts (Optional[DebugArtifacts]): artifacts object to fill
                during compilation
            show_mlir (bool): whether or not to show MLIR during the compilation
            use_virtual_lib (bool): set to True to use the so called virtual lib
                simulating FHE computation. Defaults to False
            p_error (Optional[float]): probability of error of a single PBS
            global_p_error (Optional[float]): probability of error of the full circuit. Not
                simulated by the VL, i.e., taken as 0
            verbose_compilation (bool): whether to show compilation information

        Returns:
            Circuit: the compiled Circuit.

        """
        self.check_model_is_fitted()

        _tensor_tree_predict_proxy, parameters_mapping = generate_proxy_function(
            self._tensor_tree_predict, ["inputs"]
        )

        # Return a numpy array, because compile method supports only arrays
        X = check_array_and_assert(X)

        X = self.quantize_input(X)
        compiler = Compiler(
            _tensor_tree_predict_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
        )

        # Don't let the user shoot in her foot, by having p_error or global_p_error set in both
        # configuration and in direct arguments
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Find the right way to set parameters for compiler, depending on the way we want to default
        p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

        self.fhe_circuit = compiler.compile(
            (sample.reshape(1, sample.shape[0]) for sample in X),
            configuration=configuration,
            artifacts=compilation_artifacts,
            show_mlir=show_mlir,
            virtual=use_virtual_lib,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose_compilation,
        )

        output_graph = self.fhe_circuit.graph.ordered_outputs()
        assert_true(
            len(output_graph) == 1,
            "graph has too many outputs",
        )
        dtype_output = output_graph[0].output.dtype
        assert_true(
            isinstance(dtype_output, Integer),
            f"output is {dtype_output} but an Integer is expected.",
        )

        return self.fhe_circuit

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        # Sum all tree outputs.
        # Remove the sum once we handle multi-precision circuits
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/451
        y_preds = numpy.sum(y_preds, axis=-1)
        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")
        return y_preds


class BaseTreeRegressorMixin(BaseTreeEstimatorMixin, sklearn.base.RegressorMixin):
    """Mixin class for tree-based regressors.

    This class is used to create a tree-based regressor class that inherits from
    sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method
    for regressors.
    """


class BaseTreeClassifierMixin(BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin):
    """Mixin class for tree-based classifiers.

    This class is used to create a tree-based classifier class that inherits from
    sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method
    for classifiers.

    Additionally, this class adjusts some of the tree-based base class's methods in order to make
    them compliant with classification workflows.
    """

    classes_: numpy.ndarray
    n_classes_: int
    class_mapping_: Optional[Dict[int, int]]

    def fit(self, X, y: numpy.ndarray, **kwargs) -> Any:
        """Fit the tree-based estimator.

        Args:
            X : training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): The target data.
            **kwargs: args for super().fit

        Returns:
            Any: The fitted model.
        """
        multi_output = isinstance(y[0], list) if isinstance(y, list) else len(y.shape) > 1
        X, y = check_X_y_and_assert(X, y, multi_output=multi_output)

        self.classes_ = numpy.unique(y)

        # Register the number of classes
        self.n_classes_ = len(self.classes_)

        # Make sure y contains at least two classes.
        assert_true(self.n_classes_ > 1, "You must provide at least 2 classes in y.")

        # Keep track of the classes' order if they are not sorted
        self.class_mapping_ = None
        if not numpy.array_equal(numpy.arange(len(self.classes_)), self.classes_):
            self.class_mapping_ = dict(enumerate(self.classes_))

        return super().fit(X, y, **kwargs)

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict the class with highest probability.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        self.check_model_is_fitted()

        X = check_array_and_assert(X)

        # Compute the predicted probabilities
        y_preds = self.predict_proba(X, execute_in_fhe=execute_in_fhe)

        # Retrieve the class with the highest probability
        y_preds = numpy.argmax(y_preds, axis=1)

        # Order the output classes in order to be consistent with Scikit-learn's model
        if self.class_mapping_ is not None:
            y_preds = numpy.array([self.class_mapping_[y_pred] for y_pred in y_preds])

        return y_preds

    def predict_proba(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict the probability.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.

        Returns:
            numpy.ndarray: The predicted probabilities.
        """
        self.check_model_is_fitted()

        return super().predict(X, execute_in_fhe=execute_in_fhe)


# pylint: disable=invalid-name,too-many-instance-attributes
class SklearnLinearModelMixin(BaseEstimator, sklearn.base.BaseEstimator):
    """A Mixin class for sklearn linear models with FHE.

    This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's
    `get_params` and `set_params` methods.
    """

    n_bits: Union[int, Dict[str, int]]
    sklearn_alg: Callable[..., sklearn.base.BaseEstimator]
    random_state: Optional[Union[numpy.random.RandomState, int]] = None  # pylint: disable=no-member
    _weight_quantizer: UniformQuantizer
    _output_scale: numpy.float64
    _output_zero_point: int
    _q_weights: numpy.ndarray
    _q_bias: numpy.ndarray

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
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

        super().__init__()

    @property
    def _is_fitted(self):
        return (
            self.input_quantizers
            and self.post_processing_params.get("output_scale", None) is not None
            and self.post_processing_params.get("output_zero_point", None) is not None
        )

    def fit(self, X, y: numpy.ndarray, *args, **kwargs) -> Any:
        """Fit the FHE linear model.

        Args:
            X : Training data
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
        # Copy X
        X = copy.deepcopy(X)

        # LinearRegression handles multi-labels data
        multi_output = isinstance(y[0], list) if isinstance(y, list) else len(y.shape) > 1
        X, y = check_X_y_and_assert(X, y, multi_output=multi_output)

        # Retrieve sklearn's init parameters
        params = self.get_sklearn_params()  # type: ignore

        # Initialize the sklearn model
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(X, y, *args, **kwargs)

        # This workaround makes linear regressors be able to fit with fit_intercept set to False.
        # This needs to be removed once HummingBird's latest version is integrated in Concrete-ML
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1610
        if not self.fit_intercept:
            self.sklearn_model.intercept_ = numpy.array(self.sklearn_model.intercept_)

        # These models are not natively supported by Hummingbird
        # The trick is to hide their type to Hummingbird, it should be removed once HummingBird's
        # latest version is integrated in Concrete-ML
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2792
        if self.sklearn_alg in {
            sklearn.linear_model.Lasso,
            sklearn.linear_model.Ridge,
            sklearn.linear_model.ElasticNet,
        }:
            self.sklearn_model.__class__ = sklearn.linear_model.LinearRegression
            self.n_jobs = None
            self.sklearn_model.n_jobs = None

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
        self._update_post_processing_params()

        # Quantize the bias using the matmul's scale and zero-point, such that
        # q_bias = round((1/S)*bias + Z)
        # Contrary to the QuantizedGemm operator which handles the bias term as a floating point
        # (and thus fusing it with following TLUs), we need to quantize the bias term so that it
        # matches the same range of values as the matmul's outputs.
        self._q_bias = numpy.round(
            self.sklearn_model.intercept_ / self._output_scale + self._output_zero_point
        ).astype(numpy.int64)

        return self

    def fit_benchmark(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        *args,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Fit the sklearn linear model and the FHE linear model.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            random_state (Optional[Union[int, numpy.random.RandomState, None]]):
                The random state. Defaults to None.
            *args: The arguments to pass to the sklearn linear model.
                or not (False). Default to False.
            *args: Arguments for super().fit
            **kwargs: Keyword arguments for super().fit

        Returns:
            Tuple[SklearnLinearModelMixin, sklearn.linear_model.LinearRegression]:
                The FHE and sklearn LinearRegression.
        """
        # Retrieve sklearn's init parameters
        params = self.get_sklearn_params()  # type: ignore

        # Make sure the random_state is set or both algorithms will diverge
        # due to randomness in the training.
        if "random_state" in params:
            if random_state is not None:
                params["random_state"] = random_state
            elif self.random_state is not None:
                params["random_state"] = self.random_state
            else:
                params["random_state"] = numpy.random.randint(0, 2**15)

        # Train the sklearn model without X quantized
        sklearn_model = self.sklearn_alg(**params)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        self.set_params(n_bits=self.n_bits, **params)

        self.fit(X, y, *args, **kwargs)

        return self, sklearn_model

    def compile(
        self,
        X: numpy.ndarray,
        configuration: Optional[Configuration] = None,
        compilation_artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose_compilation: bool = False,
    ) -> Circuit:
        """Compile the FHE linear model.

        Args:
            X (numpy.ndarray): The input data.
            configuration (Optional[Configuration]): Configuration object
                to use during compilation
            compilation_artifacts (Optional[DebugArtifacts]): Artifacts object to fill during
                compilation
            show_mlir (bool): If set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.
            use_virtual_lib (bool): Whether to compile using the virtual library that allows higher
                bitwidths with simulated FHE computation. Defaults to False
            p_error (Optional[float]): Probability of error of a single PBS
            global_p_error (Optional[float]): probability of error of the full circuit. Not
                simulated by the VL, i.e., taken as 0
            verbose_compilation (bool): whether to show compilation information

        Returns:
            Circuit: The compiled Circuit.

        """

        def inference_to_compile(q_X: numpy.ndarray) -> numpy.ndarray:
            """Compile the circuit in FHE using only the inputs as parameters.

            Args:
                q_X (numpy.ndarray): The quantized input data

            Returns:
                numpy.ndarray: The circuit's outputs.
            """
            return self._inference(
                q_X, self._q_weights, self._q_bias, self._weight_quantizer.zero_point
            )

        self.check_model_is_fitted()

        # Cast pandas, list or torch to numpy
        X = check_array_and_assert(X)

        # Quantize the inputs
        q_X = self.quantize_input(X)

        # Make the input set an generator yielding values with proper dimensions
        inputset = _get_inputset_generator((q_X,), self.input_quantizers)

        # Instantiate the compiler on the linear regression's inference
        compiler = Compiler(inference_to_compile, {"q_X": "encrypted"})

        # Don't let the user shoot in her foot, by having p_error or global_p_error set in both
        # configuration and in direct arguments
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Find the right way to set parameters for compiler, depending on the way we want to default
        p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

        # Compile on the input set
        self.fhe_circuit = compiler.compile(
            inputset,
            configuration=configuration,
            artifacts=compilation_artifacts,
            show_mlir=show_mlir,
            virtual=use_virtual_lib,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose_compilation,
        )

        return self.fhe_circuit

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict on user data.

        Predict on user data using either the quantized clear model,
        implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit

        Args:
            X (numpy.ndarray): The input data
            execute_in_fhe (bool): Whether to execute the inference in FHE

        Returns:
            numpy.ndarray: The prediction as ordinals
        """
        self.check_model_is_fitted()

        X = check_array_and_assert(X)

        # Quantize the inputs
        q_X = self.quantize_input(X)

        # If the predictions are executed in FHE, we call the circuit on each sample
        if execute_in_fhe:
            self.check_model_is_compiled()

            q_y_preds_list = []
            for q_X_i in q_X:
                # Ignore mypy issue as `check_model_is_compiled` makes sure that fhe_circuit is
                # not None
                q_y_pred_i = self.fhe_circuit.encrypt_run_decrypt(  # type: ignore[union-attr]
                    q_X_i.reshape(1, q_X_i.shape[0])
                )
                q_y_preds_list.append(q_y_pred_i[0])

            q_y_preds = numpy.array(q_y_preds_list)

        # If the predictions are not executed in FHE, we only need to call the inference
        else:
            q_y_preds = self._inference(
                q_X, self._q_weights, self._q_bias, self._weight_quantizer.zero_point
            )

        # Dequantize the outputs
        y_preds = self.dequantize_output(q_y_preds)

        return y_preds

    def quantize_input(self, X: numpy.ndarray):
        self.check_model_is_fitted()
        return self.input_quantizers[0].quant(X)

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()

        # Retrieve the post-processing parameters
        output_scale = self.post_processing_params["output_scale"]
        output_zero_point = self.post_processing_params["output_zero_point"]

        # Dequantize the output.
        # Since the matmul and the bias both use the same scale and zero-points, we obtain that
        # y = S*(q_y - 2*Z)
        y_preds = output_scale * (q_y_preds - 2 * output_zero_point)
        return y_preds

    def _set_onnx_model(self, test_input: numpy.ndarray):
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """
        self.onnx_model_ = hb_convert(
            self.sklearn_model,
            backend="onnx",
            test_input=test_input,
            extra_config={"onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT},
        ).model

        self.clean_graph()

    def clean_graph(self):
        """Clean the graph of the onnx model.

        This will remove the Cast node in the model's onnx.graph since they have no use in
        quantized or FHE models.
        """
        remove_node_types(onnx_model=self.onnx_model_, op_types_to_remove=["Cast"])

    def _update_post_processing_params(self):
        """Update the post processing parameters."""
        self.post_processing_params = {
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
        }

    @staticmethod
    def _inference(
        q_x: numpy.ndarray,
        q_weights: numpy.ndarray,
        q_bias: numpy.ndarray,
        weight_zp: numpy.ndarray,
    ) -> numpy.ndarray:
        """Execute a linear inference.

        Args:
            q_x (numpy.ndarray): The quantized input data
            q_weights (numpy.ndarray): The quantized weights
            q_bias (numpy.ndarray): The quantized bias
            weight_zp (int): Zero-point use for quantizing the weights

        Returns:
            numpy.ndarray: The predicted values.
        """
        # Quantizing weights and inputs makes an additional term appear in the inference function
        y_pred = q_x @ q_weights - weight_zp * numpy.sum(q_x, axis=1, keepdims=True)
        y_pred += q_bias
        return y_pred


class SklearnLinearRegressorMixin(SklearnLinearModelMixin, sklearn.base.RegressorMixin):
    """A Mixin class for sklearn linear regressors with FHE.

    This class is used to create a linear regressor class that inherits from
    sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method
    for regressors.
    """


class SklearnLinearClassifierMixin(SklearnLinearModelMixin, sklearn.base.ClassifierMixin):
    """A Mixin class for sklearn linear classifiers with FHE.

    This class is used to create a linear classifier class that inherits from
    sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method
    for classifiers.

    Additionally, this class adjusts some of the tree-based base class's methods in order to make
    them compliant with classification workflows.
    """

    def post_processing(self, y_preds: numpy.ndarray):
        # If the predictions only has one dimension (i.e. binary classification problem), apply the
        # sigmoid operator
        if y_preds.shape[1] == 1:
            y_preds = numpy_sigmoid(y_preds)[0]

            # Transform in a 2d array where [1-p, p] is the output as scikit-learn only outputs 1
            # value when considering 2 classes
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)

        # Else, apply the softmax operator
        else:
            y_preds = numpy_softmax(y_preds)[0]

        return y_preds

    def decision_function(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict confidence scores for samples.

        Args:
            X (numpy.ndarray): Samples to predict.
            execute_in_fhe (bool): If True, the inference will be executed in FHE. Default to False.

        Returns:
            numpy.ndarray: Confidence scores for samples.
        """
        y_preds = super().predict(X, execute_in_fhe=execute_in_fhe)
        return y_preds

    def predict_proba(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict class probabilities for samples.

        Args:
            X (numpy.ndarray): Samples to predict.
            execute_in_fhe (bool): If True, the inference will be executed in FHE. Default to False.

        Returns:
            numpy.ndarray: Class probabilities for samples.
        """
        y_preds = self.decision_function(X, execute_in_fhe=execute_in_fhe)
        y_preds = self.post_processing(y_preds)
        return y_preds

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict on user data.

        Predict on user data using either the quantized clear model, implemented with tensors, or,
        if execute_in_fhe is set, using the compiled FHE circuit.

        Args:
            X (numpy.ndarray): Samples to predict.
            execute_in_fhe (bool): If True, the inference will be executed in FHE. Default to False.

        Returns:
            numpy.ndarray: The prediction as ordinals.
        """
        y_preds = self.predict_proba(X, execute_in_fhe=execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds

    def clean_graph(self):
        """Clean the graph of the onnx model.

        Any operators following gemm, including the sigmoid, softmax and argmax operators, are
        removed from the graph. They will be executed in clear in the post-processing method.
        """
        clean_graph_after_node_op_type(self.onnx_model_, node_op_type="Gemm")
        super().clean_graph()


def get_sklearn_models():
    """Return the list of available models in Concrete-ML.

    Returns:
        the lists of models in Concrete-ML
    """

    # Import anything in sklearn, just to force the import, to populate _ALL_SKLEARN_MODELS list
    # pylint: disable=unused-import, import-outside-toplevel, cyclic-import
    from ..sklearn import LinearRegression  # noqa: F401

    # We return sorted lists such that it is ordered, to avoid notably issues when it is used
    # in @pytest.mark.parametrize
    ans = {
        "all": sorted(list(_ALL_SKLEARN_MODELS), key=lambda m: m.__name__),
        "linear": sorted(list(_LINEAR_MODELS), key=lambda m: m.__name__),
        "tree": sorted(list(_TREE_MODELS), key=lambda m: m.__name__),
        "neural_net": sorted(list(_NEURALNET_MODELS), key=lambda m: m.__name__),
    }
    return ans


def _filter_models(prelist, classifier: bool, regressor: bool, str_in_class_name: str = None):
    """Return the models which are in prelist and follow (classifier, regressor) conditions.

    Args:
        prelist: list of models
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (str): if not None, only return models with this as a substring in the
            class name

    Returns:
        the sublist which fulfills the (classifier, regressor, str_in_class_name) conditions.

    """
    assert_true(classifier or regressor, "Please set at least one option")

    answer = []

    if classifier:
        answer += [m for m in prelist if is_classifier_or_partial_classifier(m)]

    if regressor:
        answer += [m for m in prelist if is_regressor_or_partial_regressor(m)]

    if str_in_class_name is not None:
        answer = [m for m in answer if str_in_class_name in m.__name__]

    # We return a sorted list such that it is ordered, to avoid notably issues when it is used
    # in @pytest.mark.parametrize
    return sorted(answer, key=lambda m: m.__name__)


def get_sklearn_linear_models(
    classifier: bool = True, regressor: bool = True, str_in_class_name: str = None
):
    """Return the list of available linear models in Concrete-ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (str): if not None, only return models with this as a substring in the
            class name

    Returns:
        the lists of linear models in Concrete-ML
    """
    prelist = get_sklearn_models()["linear"]
    return _filter_models(prelist, classifier, regressor, str_in_class_name)


def get_sklearn_tree_models(
    classifier: bool = True, regressor: bool = True, str_in_class_name: str = None
):
    """Return the list of available tree models in Concrete-ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (str): if not None, only return models with this as a substring in the
            class name

    Returns:
        the lists of tree models in Concrete-ML
    """
    prelist = get_sklearn_models()["tree"]
    return _filter_models(prelist, classifier, regressor, str_in_class_name)


def get_sklearn_neural_net_models(
    classifier: bool = True, regressor: bool = True, str_in_class_name: str = None
):
    """Return the list of available neural net models in Concrete-ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (str): if not None, only return models with this as a substring in the
            class name

    Returns:
        the lists of neural net models in Concrete-ML
    """
    prelist = get_sklearn_models()["neural_net"]
    return _filter_models(prelist, classifier, regressor, str_in_class_name)
