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

import numpy
import onnx
import sklearn
import torch
from brevitas.export import BrevitasONNXManager
from concrete.numpy.compilation.artifacts import DebugArtifacts
from concrete.numpy.compilation.circuit import Circuit
from concrete.numpy.compilation.compiler import Compiler
from concrete.numpy.compilation.configuration import Configuration
from concrete.numpy.dtypes.integer import Integer

from concrete.ml.quantization.quantized_module import QuantizedModule, _get_inputset_generator
from concrete.ml.quantization.quantizers import UniformQuantizer

from ..common.check_inputs import check_array_and_assert, check_X_y_and_assert
from ..common.debugging.custom_assert import assert_true
from ..common.utils import generate_proxy_function, manage_parameters_for_pbs_errors
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from ..onnx.onnx_model_manipulations import clean_graph_after_node_op_type, remove_node_types

# The sigmoid and softmax functions are already defined in the ONNX module and thus are imported
# here in order to avoid duplicating them.
from ..onnx.ops_impl import numpy_sigmoid, numpy_softmax
from ..quantization import PostTrainingQATImporter, QuantizedArray, get_n_bits_dict
from ..torch import NumpyModule
from .protocols import Quantizer
from .tree_to_numpy import Task, tree_to_numpy

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


class QuantizedTorchEstimatorMixin:
    """Mixin that provides quantization for a torch module and follows the Estimator API.

    This class should be mixed in with another that provides the full Estimator API. This class
    only provides modifiers for .fit() (with quantization) and .predict() (optionally in FHE)
    """

    post_processing_params: Dict[str, Any] = {}
    _is_a_public_cml_model = False

    def __init_subclass__(cls):
        for klass in cls.__mro__:

            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
                _NEURALNET_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(
        self,
    ):
        # The quantized module variable appends "_" so that it is not registered as a sklearn
        # parameter. Only training parameters should register, to enable easy cloning of untrained
        # estimator
        self.quantized_module_ = None
        self._onnx_model_ = None

    @property
    @abstractmethod
    def base_estimator_type(self):
        """Get the sklearn estimator that should be trained by the child class."""

    def get_params_for_benchmark(self):
        """Get the parameters to instantiate the sklearn estimator trained by the child class.

        Returns:
            params (dict): dictionary with parameters that will initialize a new Estimator
        """
        return self.get_params()

    @property
    def input_quantizers(self) -> List[Quantizer]:
        """Get the input quantizers.

        Returns:
            List[Quantizer]: the input quantizers
        """
        return self.quantized_module_.input_quantizers

    @input_quantizers.setter
    def input_quantizers(self, value: List[Quantizer]):
        """Set the input quantizers.

        Args:
            value (List[Quantizer]): the input quantizers
        """
        self.quantized_module_ = QuantizedModule()
        self.quantized_module_.input_quantizers = value

    @property
    @abstractmethod
    def base_module_to_compile(self):
        """Get the Torch module that should be compiled to FHE."""

    @property
    @abstractmethod
    def n_bits_quant(self):
        """Get the number of quantization bits."""

    @property
    def onnx_model(self):
        """Get the ONNX model.

        .. # noqa: DAR201

        Returns:
           _onnx_model_ (onnx.ModelProto): the ONNX model
        """
        return self._onnx_model_

    @property
    def quantize_input(self) -> Callable:
        """Get the input quantization function.

        Returns:
            Callable : function that quantizes the input
        """
        assert self.quantized_module_ is not None
        return self.quantized_module_.quantize_input

    @property
    def fhe_circuit(self) -> Circuit:
        """Get the FHE circuit.

        Returns:
            Circuit: the FHE circuit
        """
        return self.quantized_module_.fhe_circuit

    @fhe_circuit.setter
    def fhe_circuit(self, value: Circuit):
        """Set the FHE circuit.

        Args:
            value (Circuit): the FHE circuit
        """
        self.quantized_module_.fhe_circuit = value

    @property
    def output_quantizers(self) -> List[QuantizedArray]:
        """Get the input quantizers.

        Returns:
            List[QuantizedArray]: the input quantizers
        """
        return self.quantized_module_.output_quantizers

    @output_quantizers.setter
    def output_quantizers(self, value: List[QuantizedArray]):
        """Set the input quantizers.

        Args:
            value (List[QuantizedArray]): the input quantizers
        """
        self.quantized_module_.output_quantizers = value

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Post-processing the output.

        Args:
            y_preds (numpy.ndarray): the output to post-process

        Raises:
            ValueError: if unknown post-processing function

        Returns:
            numpy.ndarray: the post-processed output
        """
        y_preds = self.quantized_module_.dequantize_output(y_preds)

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
            global_p_error (Optional[float]): probability of error of the full circuit
            verbose_compilation (bool): whether to show compilation information

        Returns:
            Circuit: the compiled Circuit.

        Raises:
            ValueError: if called before the model is trained
        """
        if self.quantized_module_ is None:
            raise ValueError(
                "The classifier needs to be calibrated before compilation,"
                " please call .fit() first!"
            )

        # Quantize the compilation input set using the quantization parameters computed in .fit()
        quantized_numpy_inputset = self.quantized_module_.quantize_input(X)

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
        self.quantized_module_ = None
        X, y = check_X_y_and_assert(X, y, multi_output=y.size > 1)

        # Call skorch fit that will train the network
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

        self._onnx_model_ = numpy_model.onnx_model

        # Set the quantization bits for import

        # Note that the ONNXConverter will use a default value for network input bits
        # Furthermore, Brevitas ONNX contains bitwidths in the ONNX file
        # which override the bitwidth that we pass here

        # Thus, this parameter, set by the inheriting classes, such as NeuralNetClassifier
        # is only used to check consistency during import (onnx file vs import)
        n_bits = self.n_bits_quant

        # Import the quantization aware trained model
        qat_model = PostTrainingQATImporter(n_bits, numpy_model, is_signed=True)

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

        Raises:
            ValueError: if the estimator was not yet trained or compiled
        """
        if execute_in_fhe:
            if self.quantized_module_ is None:
                raise ValueError(
                    "The classifier needs to be calibrated before compilation,"
                    " please call .fit() first!"
                )
            if not self.quantized_module_.is_compiled:
                raise ValueError(
                    "The classifier is not yet compiled to FHE, please call .compile() first"
                )

            # Run over each element of X individually and aggregate predictions in a vector
            if X.ndim == 1:
                X = X.reshape((1, -1))
            X = check_array_and_assert(X)
            y_pred = None
            for idx, x in enumerate(X):
                q_x = self.quantized_module_.quantize_input(x).reshape(1, -1)
                q_pred = self.quantized_module_.forward_fhe.encrypt_run_decrypt(q_x)
                if y_pred is None:
                    assert_true(
                        isinstance(q_pred, numpy.ndarray),
                        f"bad type {q_pred}, expected np.ndarray",
                    )
                    # pylint is lost: Instance of 'tuple' has no 'size' member (no-member)
                    # because it doesn't understand the Union in encrypt_run_decrypt
                    # pylint: disable=no-member
                    y_pred = numpy.zeros((X.shape[0], q_pred.size), numpy.float32)
                    # pylint: enable=no-member
                y_pred[idx, :] = q_pred
            y_pred = self.post_processing(y_pred)
            return y_pred

        # For prediction in the clear we call the super class which, in turn,
        # will end up calling .infer of this class
        return super().predict_proba(X)

    # pylint: enable=arguments-differ

    def fit_benchmark(self, X: numpy.ndarray, y: numpy.ndarray, *args, **kwargs) -> Tuple[Any, Any]:
        """Fit the quantized estimator and return reference estimator.

        This function returns both the quantized estimator (itself),
        but also a wrapper around the non-quantized trained NN. This is useful in order
        to compare performance between the quantized and fp32 versions of the classifier

        Args:
            X : training data
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
            y (numpy.ndarray): labels associated with training data
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.

        Returns:
            self: the trained quantized estimator
            fp32_model: trained raw (fp32) wrapped NN estimator
        """

        self.fit(X, y, *args, **kwargs)

        # Create a skorch estimator with the same training parameters as this one
        # Follow sklearn.base.clone: copy.deepcopy parameters obtained with get_params()
        # and pass them to the constructor

        # sklearn docs: "Clone does a deep copy of the model in an estimator without actually
        # copying  attached data. It returns a new estimator with the same parameters
        # that has not been fitted on any data."
        # see: https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        new_object_params = self.get_params_for_benchmark()

        for name, param in new_object_params.items():
            new_object_params[name] = copy.deepcopy(param)

        klass = self.base_estimator_type
        module_copy = copy.deepcopy(self.base_module_to_compile)

        # Construct with the fp32 network already trained for this quantized estimator
        # Need to remove the `module` parameter as we pass the trained instance here
        # This key is only present for NeuralNetClassifiers that don't fix the module type
        # Else this key may be removed already, e.g. by the FixedTypeSkorchNeuralNet
        if "module" in new_object_params:
            new_object_params.pop("module")
        fp32_model = klass(module_copy, **new_object_params)

        # Don't fit the new estimator, it is already trained. We just need to call initialize() to
        # signal to the skorch estimator that it is already trained
        fp32_model.initialize()

        return self, fp32_model


class BaseTreeEstimatorMixin(sklearn.base.BaseEstimator):
    """Mixin class for tree-based estimators.

    A place to share methods that are used on all tree-based estimators.
    """

    n_bits: int
    input_quantizers: List[UniformQuantizer]
    output_quantizers: List[UniformQuantizer]
    init_args: Dict[str, Any]
    random_state: Optional[Union[numpy.random.RandomState, int]] = None  # pylint: disable=no-member
    sklearn_alg: Callable[..., sklearn.base.BaseEstimator]
    sklearn_model: sklearn.base.BaseEstimator
    _tensor_tree_predict: Optional[Callable]
    framework: str
    fhe_circuit: Optional[Circuit] = None
    _onnx_model_: Optional[onnx.ModelProto] = None
    _is_a_public_cml_model: bool = False

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
                _TREE_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, n_bits: int):
        """Initialize the TreeBasedEstimatorMixin.

        Args:
            n_bits (int): number of bits used for quantization
        """
        self.n_bits = n_bits
        self.input_quantizers = []
        self.output_quantizers = []
        self.fhe_circuit = None
        self._onnx_model_ = None
        self.post_processing_params: Dict[str, Any] = {}

    @property
    def onnx_model(self) -> onnx.ModelProto:
        """Get the ONNX model.

        .. # noqa: DAR201

        Returns:
           onnx.ModelProto: the ONNX model
        """
        return self._onnx_model_

    def quantize_input(self, X: numpy.ndarray):
        """Quantize the input.

        Args:
            X (numpy.ndarray): the input

        Returns:
            the quantized input
        """
        qX = numpy.zeros_like(X)
        # Quantize using the learned quantization parameters for each feature
        for i, q_x_ in enumerate(self.input_quantizers):
            qX[:, i] = q_x_.quant(X[:, i])
        return qX.astype(numpy.int32)

    def dequantize_output(self, y_preds: numpy.ndarray):
        """Dequantize the integer predictions.

        Args:
            y_preds (numpy.ndarray): the predictions

        Returns:
            the dequantized predictions
        """
        # mypy
        assert self.output_quantizers is not None
        y_preds = self.output_quantizers[0].dequant(y_preds)
        return y_preds

    def _update_post_processing_params(self):
        """Update the post processing parameters."""
        self.post_processing_params = {}

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

        params = self.get_params()  # type: ignore
        params.pop("n_bits", None)

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
        # Check that self.fhe_tree is not None
        # mypy
        assert self.fhe_circuit is not None, (
            f"You must call {self.compile.__name__} "
            f"before calling {self.predict.__name__} with execute_in_fhe=True."
        )
        # mypy
        assert self.fhe_circuit is not None

        y_preds = []
        for qX_i in qX:
            # expected x shape is (n_features, n_samples)
            fhe_pred = self.fhe_circuit.encrypt_run_decrypt(
                qX_i.astype(numpy.uint8).reshape(1, qX_i.shape[0])
            )
            y_preds.append(fhe_pred)
        y_preds_array = numpy.concatenate(y_preds, axis=-1)

        return y_preds_array

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
            global_p_error (Optional[float]): probability of error of the full circuit
            verbose_compilation (bool): whether to show compilation information

        Returns:
            Circuit: the compiled Circuit.

        """
        # Make sure that self.tree_predict is not None
        assert_true(
            self._tensor_tree_predict is not None, "You must fit the model before compiling it."
        )

        # mypy bug fix
        assert self._tensor_tree_predict is not None
        _tensor_tree_predict_proxy, parameters_mapping = generate_proxy_function(
            self._tensor_tree_predict, ["inputs"]
        )

        X = self.quantize_input(X)
        compiler = Compiler(
            _tensor_tree_predict_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
        )

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
        # mypy
        assert self.fhe_circuit is not None
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


class BaseTreeRegressorMixin(BaseTreeEstimatorMixin, sklearn.base.RegressorMixin):
    """Mixin class for tree-based regressors.

    A place to share methods that are used on all tree-based regressors.
    """

    _is_a_public_cml_model: bool = False

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
                _TREE_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

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
        X, y = check_X_y_and_assert(X, y, multi_output=y.size > 1)

        # mypy
        assert self.n_bits is not None

        qX = numpy.zeros_like(X)
        self.input_quantizers = []

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i]).quantizer
            self.input_quantizers.append(q_x_)
            qX[:, i] = q_x_.quant(X[:, i])

        # Initialize the sklearn model
        params = self.get_params()
        params.pop("n_bits", None)

        self.sklearn_model = self.sklearn_alg(**params)

        self.sklearn_model.fit(qX, y, **kwargs)
        self._update_post_processing_params()

        # Tree ensemble inference to numpy
        self._tensor_tree_predict, self.output_quantizers, self._onnx_model_ = tree_to_numpy(
            self.sklearn_model,
            qX[:1],
            framework=self.framework,
            output_n_bits=self.n_bits,
            task=Task.REGRESSION,
        )
        return self

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        y_preds = self.dequantize_output(y_preds)

        # Sum all tree outputs.
        # FIXME Remove this once #451 is done :
        # https://github.com/zama-ai/concrete-ml-internal/issues/451
        y_preds = numpy.sum(y_preds, axis=0)
        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")

        # FIXME transpose workaround see #931
        # https://github.com/zama-ai/concrete-ml-internal/issues/931
        y_preds = numpy.transpose(y_preds)
        return y_preds

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict the probability.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.

        Returns:
            numpy.ndarray: The predicted probabilities.
        """

        X = check_array_and_assert(X)

        # mypy
        assert self._tensor_tree_predict is not None
        qX = self.quantize_input(X)
        if execute_in_fhe:
            y_preds = self._execute_in_fhe(qX)
        else:
            y_preds = self._tensor_tree_predict(qX)[0]
        y_preds = self.post_processing(y_preds)
        return y_preds


class BaseTreeClassifierMixin(BaseTreeEstimatorMixin, sklearn.base.ClassifierMixin):
    """Mixin class for tree-based classifiers.

    A place to share methods that are used on all tree-based classifiers.
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
        X, y = check_X_y_and_assert(X, y, multi_output=y.size > 1)

        self.classes_ = numpy.unique(y)

        # Register the number of classes
        self.n_classes_ = len(self.classes_)

        # Make sure y contains at least two classes.
        assert_true(self.n_classes_ > 1, "You must provide at least 2 classes in y.")

        self.class_mapping_ = None
        if not numpy.array_equal(numpy.arange(len(self.classes_)), self.classes_):
            self.class_mapping_ = dict(enumerate(self.classes_))

        # mypy
        assert self.n_bits is not None

        qX = numpy.zeros_like(X)
        self.input_quantizers = []

        # Quantization of each feature in X
        for i in range(X.shape[1]):
            q_x_ = QuantizedArray(n_bits=self.n_bits, values=X[:, i]).quantizer
            self.input_quantizers.append(q_x_)
            qX[:, i] = q_x_.quant(X[:, i])

        # Initialize the sklearn model
        params = self.get_params()
        params.pop("n_bits", None)

        self.sklearn_model = self.sklearn_alg(**params)

        self.sklearn_model.fit(qX, y, **kwargs)
        self._update_post_processing_params()

        # Tree ensemble inference to numpy
        self._tensor_tree_predict, self.output_quantizers, self._onnx_model_ = tree_to_numpy(
            self.sklearn_model,
            qX[:1],
            framework=self.framework,
            output_n_bits=self.n_bits,
            task=Task.CLASSIFICATION,
        )
        return self

    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        """Predict the class with highest probability.

        Args:
            X (numpy.ndarray): The input data.
            execute_in_fhe (bool): Whether to execute in FHE. Defaults to False.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        X = check_array_and_assert(X)

        y_preds = self.predict_proba(X, execute_in_fhe=execute_in_fhe)
        y_preds = numpy.argmax(y_preds, axis=1)
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
        X = check_array_and_assert(X)

        # mypy
        assert self._tensor_tree_predict is not None
        qX = self.quantize_input(X)
        if execute_in_fhe:
            y_preds = self._execute_in_fhe(qX)
        else:
            y_preds = self._tensor_tree_predict(qX)[0]
        y_preds = self.post_processing(y_preds)
        return y_preds

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        y_preds = self.dequantize_output(y_preds)

        # Sum all tree outputs.
        # FIXME Remove this once #451 is done :
        # https://github.com/zama-ai/concrete-ml-internal/issues/451
        y_preds = numpy.sum(y_preds, axis=0)
        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")

        # FIXME transpose workaround see #931
        # https://github.com/zama-ai/concrete-ml-internal/issues/931
        y_preds = numpy.transpose(y_preds)
        return y_preds


# pylint: disable=invalid-name,too-many-instance-attributes
class SklearnLinearModelMixin(sklearn.base.BaseEstimator):
    """A Mixin class for sklearn linear models with FHE."""

    sklearn_alg: Callable[..., sklearn.base.BaseEstimator]
    random_state: Optional[Union[numpy.random.RandomState, int]] = None  # pylint: disable=no-member

    _is_a_public_cml_model: bool = False

    def __init_subclass__(cls):
        for klass in cls.__mro__:
            # pylint: disable-next=protected-access
            if hasattr(klass, "_is_a_public_cml_model") and klass._is_a_public_cml_model:
                _LINEAR_MODELS.add(cls)
                _ALL_SKLEARN_MODELS.add(cls)

    def __init__(self, *args, n_bits: Union[int, Dict[str, int]] = 8, **kwargs):
        """Initialize the FHE linear model.

        Args:
            n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
                for n_bits, the value will be used for quantizing inputs and weights. If a dict is
                passed, then it should contain "op_inputs" and "op_weights" as keys with
                corresponding number of quantization bits so that:
                    - op_inputs : number of bits to quantize the input values
                    - op_weights: number of bits to quantize the learned parameters
                Default to 8.
            *args: The arguments to pass to the sklearn linear model.
            **kwargs: The keyword arguments to pass to the sklearn linear model.
        """
        super().__init__(*args, **kwargs)
        self.n_bits: Union[int, Dict[str, int]] = n_bits
        self.post_processing_params: Dict[str, Any] = {}
        self.fhe_circuit: Optional[Circuit] = None
        self.input_quantizers: List[UniformQuantizer] = []
        self.output_quantizers: List[UniformQuantizer] = []
        self.weight_quantizers: List[UniformQuantizer] = []
        self.onnx_model: onnx.ModelProto = None
        self._output_scale: Optional[float] = None
        self._output_zero_point: Optional[int] = None
        self._q_weights: Optional[numpy.ndarray] = None
        self._q_bias: Optional[numpy.ndarray] = None

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
        X, y = check_X_y_and_assert(X, y, multi_output=y.size > 1)

        # Retrieve sklearn's init parameters and remove the ones specific to Concrete-ML
        params = self.get_params()  # type: ignore
        params.pop("n_bits", None)

        # Initialize the sklearn model
        self.sklearn_model = self.sklearn_alg(**params)

        # Fit the sklearn model
        self.sklearn_model.fit(X, y, *args, **kwargs)

        # FIXME: Remove this when #1610 is done
        # This workaround makes linear regressors be able to fit with fit_intercept set to False.
        if not self.fit_intercept:
            self.sklearn_model.intercept_ = numpy.array(self.sklearn_model.intercept_)

        # These models are not natively supported by Hummingbird
        # The trick is to hide their type to Hummingbird
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/1473
        # Open PR to hummingbird to add support
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

        # Quantize the inputs and store the associated quantizer
        q_inputs = QuantizedArray(n_bits=n_bits["op_inputs"], values=X)
        input_quantizer = q_inputs.quantizer
        self.input_quantizers.append(input_quantizer)

        # Quantize the weights and store the associated quantizer
        # Transpose and expand are necessary in order to make sure the weight array has the correct
        # shape when calling the Gemm operator on it
        weights = self.sklearn_model.coef_.T
        q_weights = QuantizedArray(
            n_bits=n_bits["op_weights"],
            values=numpy.expand_dims(weights, axis=1) if len(weights.shape) == 1 else weights,
        )
        self._q_weights = q_weights.qvalues
        weight_quantizer = q_weights.quantizer
        self.weight_quantizers.append(weight_quantizer)

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

        # Retrieve sklearn's init parameters and remove the ones specific to Concrete-ML
        params = self.get_params()  # type: ignore
        params.pop("n_bits", None)

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
            global_p_error (Optional[float]): probability of error of the full circuit
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
            # mypy
            assert (
                self.weight_quantizers and self._q_weights is not None and self._q_bias is not None
            ), "The model is not fitted. Please run fit(...) on proper arguments first."

            return self._inference(
                q_X, self._q_weights, self._q_bias, self.weight_quantizers[0].zero_point
            )

        # mypy
        assert (
            self.input_quantizers
        ), "The model is not fitted. Please run fit(...) on proper arguments first."

        # Quantize the inputs
        q_X = self.input_quantizers[0].quant(X)

        # Make the input set an generator yielding values with proper dimensions
        inputset = _get_inputset_generator((q_X,), self.input_quantizers)

        # Instantiate the compiler on the linear regression's inference
        compiler = Compiler(inference_to_compile, {"q_X": "encrypted"})

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

        X = check_array_and_assert(X)

        # mypy
        # The model is fitted if and only if the model's input quantizer is initialized
        assert (
            self.input_quantizers
        ), "The model is not fitted. Please run fit(...) on proper arguments first."

        # Quantize the inputs
        q_X = self.quantize_input(X)

        # If the predictions are executed in FHE, we call the circuit on each sample
        if execute_in_fhe:

            # mypy
            # Make sure the model is compiled
            assert (
                self.fhe_circuit is not None
            ), "The model is not compiled. Please run compile(...) first."

            q_y_preds_list = []
            for q_X_i in q_X:
                q_y_pred_i = self.fhe_circuit.encrypt_run_decrypt(q_X_i.reshape(1, q_X_i.shape[0]))
                q_y_preds_list.append(q_y_pred_i[0])

            q_y_preds = numpy.array(q_y_preds_list)

        # If the predictions are not executed in FHE, we only need to call the inference
        else:
            # mypy
            assert (
                self.weight_quantizers and self._q_weights is not None and self._q_bias is not None
            ), "The model is not fitted. Please run fit(...) on proper arguments first."

            q_y_preds = self._inference(
                q_X, self._q_weights, self._q_bias, self.weight_quantizers[0].zero_point
            )

        # Dequantize the outputs
        y_preds = self.dequantize_output(q_y_preds)

        return y_preds

    def quantize_input(self, X: numpy.ndarray):
        """Quantize the input.

        Args:
            X (numpy.ndarray): The input to quantize

        Returns:
            numpy.ndarray: The quantized input
        """
        return self.input_quantizers[0].quant(X)

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        """Dequantize the output.

        Args:
            q_y_preds (numpy.ndarray): The quantized output to dequantize

        Returns:
            numpy.ndarray: The dequantized output
        """
        # Retrieve the post-processing parameters
        output_scale = self.post_processing_params["output_scale"]
        output_zero_point = self.post_processing_params["output_zero_point"]

        assert (
            output_scale is not None and output_zero_point is not None
        ), "The model is not fitted. Please run fit(...) on proper arguments first."

        # Dequantize the output.
        # Since the matmul and the bias both use the same scale and zero-points, we obtain that
        # y = S*(q_y - 2*Z)
        y_preds = output_scale * (q_y_preds - 2 * output_zero_point)
        return y_preds

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Post-processing the quantized output.

        For linear models, post-processing only considers a dequantization step.

        Args:
            y_preds (numpy.ndarray): The quantized outputs to post-process

        Returns:
            numpy.ndarray: The post-processed output
        """
        return self.dequantize_output(y_preds)

    def _set_onnx_model(self, test_input: numpy.ndarray):
        """Retrieve the model's ONNX graph using Hummingbird conversion.

        Args:
            test_input (numpy.ndarray): An input data used to trace the model execution.
        """

        assert_true(
            self.sklearn_model is not None,
            "The model is not fitted. Please run fit(...) on proper arguments first.",
        )

        self.onnx_model = hb_convert(
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
        remove_node_types(onnx_model=self.onnx_model, op_types_to_remove=["Cast"])

    def _update_post_processing_params(self):
        """Update the post processing parameters."""

        assert (
            self._output_scale is not None and self._output_zero_point is not None
        ), "The model is not fitted. Please run fit(...) on proper arguments first."

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


class SklearnLinearClassifierMixin(SklearnLinearModelMixin):
    """A Mixin class for sklearn linear classifiers with FHE."""

    def post_processing(self, y_preds: numpy.ndarray, already_dequantized: bool = False):
        """Post-processing the predictions.

        This step may include a dequantization of the inputs if not done previously, in particular
        within the client-server workflow.

        Args:
            y_preds (numpy.ndarray): The predictions to post-process.
            already_dequantized (bool): Wether the inputs were already dequantized or not. Default
                to False.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """

        # If y_preds were already dequantized previously, there is no need to do so once again.
        # This step is necessary for the client-server workflow as the post_processing method
        # is directly called on the quantized outputs, contrary to the base class' predict method.
        if not already_dequantized:
            y_preds = self.dequantize_output(y_preds)

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
        y_preds = self.post_processing(y_preds, already_dequantized=True)
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
        clean_graph_after_node_op_type(self.onnx_model, node_op_type="Gemm")
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


def _filter_models(prelist, classifier, regressor):
    """Return the models which are in prelist and follow (classifier, regressor) conditions.

    Args:
        prelist: list of models
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not

    Returns:
        the sublist which fullfils the (classifier, regressor) conditions.

    """
    assert_true(classifier or regressor, "Please set at least one option")

    answer = []

    if classifier:
        answer += [m for m in prelist if sklearn.base.is_classifier(m)]

    if regressor:
        answer += [m for m in prelist if sklearn.base.is_regressor(m)]

    # We return a sorted list such that it is ordered, to avoid notably issues when it is used
    # in @pytest.mark.parametrize
    return sorted(answer, key=lambda m: m.__name__)


def get_sklearn_tree_models(classifier=True, regressor=True):
    """Return the list of available tree models in Concrete-ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not

    Returns:
        the lists of tree models in Concrete-ML
    """
    prelist = get_sklearn_models()["tree"]
    return _filter_models(prelist, classifier, regressor)


def get_sklearn_neural_net_models(classifier=True, regressor=True):
    """Return the list of available neural net models in Concrete-ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not

    Returns:
        the lists of neural net models in Concrete-ML
    """
    prelist = get_sklearn_models()["neural_net"]
    return _filter_models(prelist, classifier, regressor)
