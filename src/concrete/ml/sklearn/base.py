"""Module that contains base classes for our libraries estimators."""

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import concrete.numpy as hnp
import numpy
import torch
from concrete.common.compilation.artifacts import CompilationArtifacts
from concrete.common.compilation.configuration import CompilationConfiguration
from concrete.common.data_types import Integer

from ..common.debugging.custom_assert import assert_true
from ..common.utils import generate_proxy_function
from ..quantization import PostTrainingAffineQuantization, QuantizedArray
from ..torch import NumpyModule
from ..virtual_lib import VirtualNPFHECompiler


class QuantizedTorchEstimatorMixin:
    """Mixin that provides quantization for a torch module and follows the Estimator API.

    This class should be mixed in with another that provides the full Estimator API. This class
    only provides modifiers for .fit() (with quantization) and .predict() (optionally in FHE)
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # The quantized module variable appends "_" so that it is not registered as a sklearn
        # parameter. Only training parameters should register, to enable easy cloning of un-trained
        # estimator
        self.quantized_module_ = None

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
    @abstractmethod
    def base_module_to_compile(self):
        """Get the Torch module that should be compiled to FHE."""

    @property
    @abstractmethod
    def n_bits_quant(self):
        """Get the number of quantization bits."""

    def compile(
        self,
        X: numpy.ndarray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
    ):
        """Compile the model.

        Args:
            X (numpy.ndarray): the unquantized dataset
            compilation_configuration (Optional[CompilationConfiguration]): the options for
                compilation
            compilation_artifacts (Optional[CompilationArtifacts]): artifacts object to fill
                during compilation
            show_mlir (bool): whether or not to show MLIR during the compilation
            use_virtual_lib (bool): whether to compile using the virtual library that allows higher
                bitwidths

        Raises:
            ValueError: if called before the model is trained
        """
        if self.quantized_module_ is None:
            raise ValueError(
                "The classifier needs to be calibrated before compilation,"
                " please call .fit() first!"
            )

        # Quantize the compilation input set using the quantization parameters computed in .fit()
        quantized_numpy_inputset = deepcopy(self.quantized_module_.q_inputs[0])
        quantized_numpy_inputset.update_values(X)

        # Call the compilation backend to produce the FHE inference circuit
        self.quantized_module_.compile(
            quantized_numpy_inputset,
            compilation_configuration=compilation_configuration,
            compilation_artifacts=compilation_artifacts,
            show_mlir=show_mlir,
            use_virtual_lib=use_virtual_lib,
        )

    def fit(self, X, y, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True). In addition to the
        torch training step, this method performs quantization of the trained torch model.

        Args:
            X : training data, compatible with skorch.dataset.Dataset
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
                * scipy sparse CSR matrices
                * a dictionary of the former three
                * a list/tuple of the former three
                * a Dataset
                If this doesn't work with your data, you have to pass a
                ``Dataset`` that can deal with the data.
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

        # Call skorch fit that will train the network
        super().fit(X, y, **fit_params)

        # Create corresponding numpy model
        numpy_model = NumpyModule(self.base_module_to_compile, torch.tensor(X[0, ::]))

        # Get the number of bits used in model creation (used to setup pruning)
        n_bits = self.n_bits_quant

        # Quantize with post-training static method, to have a model with integer weights
        post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model, is_signed=True)
        self.quantized_module_ = post_training_quant.quantize_module(X)
        return self

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
            y_pred = None
            for idx, x in enumerate(X):
                q_x = self.quantized_module_.quantize_input(x).reshape(1, -1)
                q_pred = self.quantized_module_.forward_fhe.run(q_x)
                if y_pred is None:
                    y_pred = numpy.zeros((X.shape[0], q_pred.size), numpy.float32)
                y_pred[idx, :] = self.quantized_module_.dequantize_output(q_pred)

            nonlin = self._get_predict_nonlinearity()
            y_pred = nonlin(torch.from_numpy(y_pred)).numpy()

            return y_pred

        # For prediction in the clear we call the super class which, in turn,
        # will end up calling .infer of this class
        return super().predict_proba(X)

    # pylint: enable=arguments-differ

    def fit_benchmark(self, X, y):
        """Fit the quantized estimator and return reference estimator.

        This function returns both the quantized estimator (itself),
        but also a wrapper around the non-quantized trained NN. This is useful in order
        to compare performance between the quantized and fp32 versions of the classifier

        Args:
            X : training data, compatible with skorch.dataset.Dataset
                By default, you should be able to pass:
                * numpy arrays
                * torch tensors
                * pandas DataFrame or Series
                * scipy sparse CSR matrices
                * a dictionary of the former three
                * a list/tuple of the former three
                * a Dataset
                If this doesn't work with your data, you have to pass a
                ``Dataset`` that can deal with the data.
            y (numpy.ndarray): labels associated with training data

        Returns:
            self: the trained quantized estimator
            fp32_model: trained raw (fp32) wrapped NN estimator
        """

        self.fit(X, y)

        # Create a skorch estimator with the same training parameters as this one
        # Follow sklearn.base.clone: deepcopy parameters obtained with get_params()
        # and pass them to the constructor

        # sklearn docs: "Clone does a deep copy of the model in an estimator without actually
        # copying  attached data. It returns a new estimator with the same parameters
        # that has not been fitted on any data."
        # see: https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        new_object_params = self.get_params_for_benchmark()

        for name, param in new_object_params.items():
            new_object_params[name] = deepcopy(param)

        klass = self.base_estimator_type
        module_copy = deepcopy(self.base_module_to_compile)

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


class BaseTreeEstimatorMixin:
    """Mixin class for tree-based estimators.

    This class is used to add functionality to tree-based estimators, such as
    the tree-based classifier.
    """

    q_x_byfeatures: List[QuantizedArray]
    n_bits: int
    q_y: QuantizedArray
    init_args: Dict[str, Any]
    random_state: Optional[Union[numpy.random.RandomState, int]]  # pylint: disable=no-member
    sklearn_alg: Any
    output_is_signed: bool
    _tensor_tree_predict: Optional[Callable]

    def __init__(self, n_bits: int):
        """Initialize the TreeBasedEstimatorMixin.

        Args:
            n_bits (int): number of bits used for quantization
        """
        self.n_bits = n_bits

    def quantize_input(self, X: numpy.ndarray):
        """Quantize the input.

        Args:
            X (numpy.ndarray): the input

        Returns:
            the quantized input
        """
        qX = numpy.zeros_like(X)
        # Quantize using the learned quantization parameters for each feature
        for i, q_x_ in enumerate(self.q_x_byfeatures):
            qX[:, i] = q_x_.update_values(X[:, i])
        return qX.astype(numpy.int32)

    @abstractmethod
    def fit(self, X, y):
        """Fit the tree-based estimator.

        Args:
            X (numpy.ndarray): the input
            y (numpy.ndarray): the labels
        """

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
        y_preds = self.predict_proba(X, execute_in_fhe=execute_in_fhe, *args, **kwargs)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the predictions.

        Args:
            y_preds (numpy.ndarray): The predictions.

        Returns:
            numpy.ndarray: The post-processed predictions.
        """
        # mypy
        assert self.q_y is not None
        y_preds = self.q_y.update_quantized_values(y_preds)
        # Sum all tree outputs.
        y_preds = numpy.sum(y_preds, axis=0)
        assert_true(y_preds.ndim == 2, "y_preds should be a 2D array")
        y_preds = numpy.transpose(y_preds)
        return y_preds

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
        assert_true(len(args) == 0, f"Unsupported *args parameters {args}")
        assert_true(len(kwargs) == 0, f"Unsupported **kwargs parameters {kwargs}")
        # mypy
        assert self._tensor_tree_predict is not None
        qX = self.quantize_input(X)
        if execute_in_fhe:
            y_preds = self._execute_in_fhe(X)
        else:
            qX = qX.transpose()
            y_preds = self._tensor_tree_predict(qX)[0]
        y_preds = self.post_processing(y_preds)
        return y_preds

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
        # Make sure the random_state is set or both algorithms will diverge
        # due to randomness in the training.
        if random_state is not None:
            self.init_args["random_state"] = random_state
        elif self.random_state is not None:
            self.init_args["random_state"] = self.random_state
        else:
            self.init_args["random_state"] = numpy.random.randint(0, 2**15)

        # Train the sklearn model without X quantized
        sklearn_model = self.sklearn_alg(**self.init_args)
        sklearn_model.fit(X, y, *args, **kwargs)

        # Train the FHE model
        self.__init__(n_bits=self.n_bits, **self.init_args)  # type: ignore
        self.fit(X, y, *args, **kwargs)
        return self, sklearn_model

    def _execute_in_fhe(self, X: numpy.ndarray) -> numpy.ndarray:
        """Execute the FHE inference on the input data.

        Args:
            X (numpy.ndarray): the input data

        Returns:
            numpy.ndarray: the prediction as ordinals
        """
        qX = self.quantize_input(X)
        # Check that self.fhe_tree is not None
        assert_true(
            self.fhe_tree is not None,
            f"You must call {self.compile.__name__} "
            f"before calling {self.predict.__name__} with execute_in_fhe=True.",
        )
        y_preds = []
        for qX_i in qX:
            # FIXME transpose workaround see #292
            # expected x shape is (n_features, n_samples)
            fhe_pred = self.fhe_tree.run(qX_i.astype(numpy.uint8).reshape(qX_i.shape[0], 1))
            y_preds.append(fhe_pred)
        y_preds_array = numpy.concatenate(y_preds, axis=-1)
        if self.output_is_signed:
            # FIXME work around for signed integers
            # see https://github.com/zama-ai/concrete-ml-internal/issues/556
            negative_idx = y_preds_array >= 2 ** (self.output_compiled_bitwidth)
            y_preds_array = numpy.where(
                negative_idx,
                y_preds_array - 2 ** (self.output_compiled_bitwidth + 1),
                y_preds_array,
            )
        return y_preds_array

    def compile(
        self,
        X: numpy.ndarray,
        compilation_configuration: Optional[CompilationConfiguration] = None,
        compilation_artifacts: Optional[CompilationArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
    ):
        """Compile the model.

        Args:
            X (numpy.ndarray): the unquantized dataset
            compilation_configuration (Optional[CompilationConfiguration]): the options for
                compilation
            compilation_artifacts (Optional[CompilationArtifacts]): artifacts object to fill
                during compilation
            show_mlir (bool): whether or not to show MLIR during the compilation
            use_virtual_lib (bool): set to True to use the so called virtual lib
                simulating FHE computation. Defaults to False.
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

        compiler_class = VirtualNPFHECompiler if use_virtual_lib else hnp.NPFHECompiler

        X = self.quantize_input(X)
        compiler = compiler_class(
            _tensor_tree_predict_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
            compilation_configuration,
            compilation_artifacts,
        )
        self.fhe_tree = compiler.compile_on_inputset(
            (sample.reshape(sample.shape[0], 1) for sample in X), show_mlir
        )

        # FIXME work around for signed integers
        # see https://github.com/zama-ai/concrete-ml-internal/issues/556
        output_op_graph = self.fhe_tree.op_graph.get_ordered_outputs()
        assert_true(
            (len(output_op_graph) == 1) and (len(output_op_graph[0].outputs) == 1),
            "op_graph has too many outputs",
        )
        assert_true(
            isinstance(dtype_output := output_op_graph[0].outputs[0].dtype, Integer),
            f"output is {dtype_output} but an Integer is expected.",
        )
        self.output_compiled_bitwidth = dtype_output.bit_width
        self.output_is_signed = dtype_output.is_signed
