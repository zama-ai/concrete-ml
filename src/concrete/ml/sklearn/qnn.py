"""Scikit-learn interface for concrete quantized neural networks."""

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.utils.prune as pruning
from concrete.common.compilation.artifacts import CompilationArtifacts
from concrete.common.compilation.configuration import CompilationConfiguration
from skorch.classifier import NeuralNetClassifier as SKNeuralNetClassifier
from torch import nn

from ..quantization import PostTrainingAffineQuantization
from ..torch import NumpyModule


class SparseQuantNNClassifier(nn.Module):
    """Sparse Quantized Neural Network classifier.

    This class implements an MLP that is compatible with FHE constraints. The weights and
    activations are quantized to low bitwidth and pruning is used to ensure accumulators do not
    surpass an user-provided accumulator bit-width. The number of classes and number of layers
    are specified by the user, as well as the breadth of the network
    """

    def __init__(
        self,
        input_dim,
        n_layers,
        n_classes,
        n_hidden_neurons_multiplier=4,
        n_w_bits=3,
        n_a_bits=3,
        n_accum_bits=7,
        activation_function=nn.ReLU,
    ):
        """Sparse Quantized Neural Network constructor.

        Args:
            input_dim: Number of dimensions of the input data
            n_layers: Number of linear layers for this network
            n_classes: Number of output classes
            n_w_bits: Number of weight bits
            n_a_bits: Number of activation and input bits
            n_accum_bits: Maximal allowed bitwidth of intermediate accumulators
            n_hidden_neurons_multiplier: A factor that is multiplied by the maximal number of
                active (non-zero weight) neurons for every layer. The maximal number of neurons in
                the worst case scenario is:
                                                                      2^n_max-1
                       max_active_neurons(n_max, n_w, n_a) = floor(---------------------)
                                                                   (2^n_w-1)*(2^n_a-1) )
                The worst case scenario for the bitwidth of the accumulator is when all weights and
                activations are maximum simultaneously. We set, for each layer, the total number of
                neurons to be:
                 n_hidden_neurons_multiplier * max_active_neurons(n_accum_bits, n_w_bits, n_a_bits)
                Through experiments, for typical distributions of weights and activations,
                the default value for n_hidden_neurons_multiplier, 4, is safe to avoid overflow.
            activation_function: a torch class that is used to construct activation functions in
                the network (e.g. torch.ReLU, torch.SELU, torch.Sigmoid, etc)

        Raises:
            ValueError: if the parameters have invalid values or the computed accumulator bitwidth
                        is zero
        """

        super().__init__()

        self.features = nn.Sequential()
        in_features = input_dim

        if input_dim <= 0:
            raise ValueError(
                f"Invalid number of input dimensions: {input_dim}, input_dim must be greater than 0"
            )

        if n_layers <= 0:
            raise ValueError(
                f"Invalid number of layers: {n_layers}, at least one intermediary layers is needed"
            )

        if n_classes < 2:
            raise ValueError(
                f"Invalid number of classes: {n_classes}, n_classes should be larger than two"
            )

        if n_w_bits <= 0 or n_a_bits <= 0:
            raise ValueError("The weight & activation quantization bitwidth cannot be less than 1")

        for idx in range(n_layers):
            out_features = (
                n_classes if idx == n_layers - 1 else input_dim * n_hidden_neurons_multiplier
            )
            layer_name = f"fc{idx}"
            layer = nn.Linear(in_features=in_features, out_features=out_features)
            self.features.add_module(layer_name, layer)
            self.features.add_module(f"act{idx}", activation_function())
            in_features = out_features

        self.n_w_bits = n_w_bits
        self.n_a_bits = n_a_bits
        self.n_accum_bits = n_accum_bits

        self.pruned_layers = set()

        self.enable_pruning()

    def max_active_neurons(self):
        """Compute the maximum number of active (non-zero weight) neurons.

        The computation is done using the quantization parameters passed to the constructor.
        Warning: With the current quantization algorithm (asymmetric) the value returned by this
        function is not guaranteed to ensure FHE compatibility. For some weight distributions,
        weights that are 0 (which are pruned weights) will not be quantized to 0.
        Therefore the total number of active quantized neurons will not be equal to
        max_active_neurons.

        Returns:
            n (int): maximum number of active neurons
        """

        return int(
            np.floor(
                (2**self.n_accum_bits - 1) / (2**self.n_w_bits - 1) / (2**self.n_a_bits - 1)
            )
        )

    def make_pruning_permanent(self):
        """Make the learned pruning permanent in the network."""
        max_neuron_connections = self.max_active_neurons()

        # Iterate over all layers that have weights (Linear ones)
        for layer in self.features:
            if not isinstance(layer, nn.Linear):
                continue

            s = layer.weight.shape
            # If this is a layer that should be pruned and is currently being pruned,
            # Make the pruning permanent
            if s[0] > max_neuron_connections and layer in self.pruned_layers:
                pruning.remove(layer, "weight")
                self.pruned_layers.remove(layer)

    def enable_pruning(self):
        """Enable pruning in the network. Pruning must be made permanent to recover pruned weights.

        Raises:
            ValueError: if the quantization parameters are invalid
        """
        max_neuron_connections = self.max_active_neurons()

        if max_neuron_connections == 0:
            raise ValueError(
                "The maximum accumulator bitwidth is too low "
                "for the quantization parameters requested. No neurons would be created in the "
                "requested configuration"
            )

        # Iterate over all layers that have weights (Linear ones)
        for layer in self.features:
            if not isinstance(layer, nn.Linear):
                continue

            s = layer.weight.shape
            # To satisfy accumulator bitwidth constraints each dot-product between an input line and
            # weight column must not exceed n_accum_bits bits. We thus prune the layer to have
            # at most max_neuron_connections non-zero weights
            if s[0] > max_neuron_connections and layer not in self.pruned_layers:
                pruning.l1_unstructured(layer, "weight", (s[0] - max_neuron_connections) * s[1])
                self.pruned_layers.add(layer)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): network input

        Returns:
            x (torch.Tensor): network prediction
        """
        return self.features(x)


class NeuralNetClassifier(SKNeuralNetClassifier):
    """Scikit-learn interface for quantized FHE compatible neural networks.

    This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator.
    It uses the skorch package to handle training and scikit-learn compatibility,
    and adds quantization and compilation functionality.

    The datatypes that are allowed for prediction by this wrapper are more restricted than
    standard scikit-learn estimators as this class needs to predict in FHE and network inference
    executor is the NumpyModule.
    """

    # FIXME: make this class accept a generic NN and not impose our SparseQuantNNClassifier
    # see https://github.com/zama-ai/concrete-ml-internal/issues/327
    def __init__(
        self,
        *args,
        criterion=torch.nn.CrossEntropyLoss,
        classes=None,
        optimizer=torch.optim.Adam,
        **kwargs,
    ):
        # A helper for users so they don't need to import torch directly
        args_to_convert_to_tensor = ["criterion__weight"]
        for arg_name in args_to_convert_to_tensor:
            if arg_name in kwargs and isinstance(kwargs[arg_name], np.ndarray):
                kwargs[arg_name] = torch.from_numpy(kwargs[arg_name]).float()

        # Note that our default optimizer is Adam which was found to be more stable when pruning
        super().__init__(
            SparseQuantNNClassifier,
            *args,
            criterion=criterion,
            classes=classes,
            optimizer=optimizer,
            **kwargs,
        )

        # The quantized module variable appends "_" so that it is not registered as a sklearn
        # parameter. Only training parameters should register, to enable easy cloning of un-trained
        # estimator
        self.quantized_module_ = None

    def compile(
        self,
        X: np.ndarray,
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
            use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
                Defaults to False.

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

    def infer(self, x, **fit_params):
        """Perform a single inference step on a batch of data.

        Args:
            x (torch.Tensor): A batch of the input data, produced by a Dataset
            **fit_params (dict) : Additional parameters passed to the ``forward`` method of
                the module and to the ``self.train_split`` call.

        Returns:
            A torch tensor with the inference results for each item in the input
        """

        # During training we infer the same as in the base class
        # Note that this supports more data types for x than we support in quantized mode
        if self.quantized_module_ is None:
            return super().infer(x, **fit_params)

        # Get a numpy array from the tensor to do quantization
        x = x.detach().cpu().numpy()

        # Once we finished the data type checks, quantize the input and perform quantized inference
        q_x = self.quantized_module_.quantize_input(x)
        q_out = self.quantized_module_(q_x)

        # Cast back to a tensor to keep a consistent API (tensor in, tensor out)
        return torch.tensor(self.quantized_module_.dequantize_output(q_out))

    def get_params(self, deep=True, **kwargs):
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
            **kwargs: any additional parameters to pass to the sklearn BaseEstimator class

        Returns:
            params : dict, Parameter names mapped to their values.
        """
        params = super().get_params(deep, **kwargs)
        # Remove the module key since our NN skorch class imposes SparseQuantNNClassifier as
        # the NN model type. Therefore, when cloning this estimator we don't need to pass
        # module again, it's passed by this class's constructor

        # FIXME: when this module becomes more generic to accept any NN, remove this
        # see
        params.pop("module")
        return params

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

        # We must remove all parameters related to the module. Skorch takes either a class or a
        # class instance for the `module` parameter. We want to pass our trained model, a class
        # instance. But for this to work, we need to remove all module related constructor params.
        # If not, skorch will instantiate a new class instance of the same type as the passed module
        # see skorch net.py NeuralNet::initialize_instance

        # sklearn docs: "Clone does a deep copy of the model in an estimator without actually
        # copying  attached data. It returns a new estimator with the same parameters
        # that has not been fitted on any data."
        # see: https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        new_object_params = self.get_params(deep=False)
        module_params = [name for name in new_object_params.keys() if "module__" in name]
        for name in module_params:
            new_object_params.pop(name)

        for name, param in new_object_params.items():
            new_object_params[name] = deepcopy(param)

        module_copy = deepcopy(self.module_)

        # Construct with the fp32 network already trained for this quantized estimator
        # The module is removed as a parameter in this class' get_params()
        fp32_model = SKNeuralNetClassifier(module_copy, **new_object_params)

        # Don't fit the new estimator, it is already trained. We just need to call initialize() to
        # signal to the skorch estimator that it is already trained
        fp32_model.initialize()

        return self, fp32_model

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
            fp32_model: trained raw (fp32) wrapped NN estimator
        """
        # Reset the quantized module since quantization is lost during refit
        # This will make the .infer() function call into the Torch nn.Module
        # Instead of the quantized module
        self.quantized_module_ = None

        # Call skorch fit that will train the network
        super().fit(X, y, **fit_params)

        # While training with pruning, a 0/1 mask is used during inference but all weights are kept
        # Once the module is trained, burn in the pruning (set layer weights to 0)
        self.module_.make_pruning_permanent()

        # Create corresponding numpy model
        numpy_model = NumpyModule(self.module_, torch.tensor(X[0, ::]))

        # Get the number of bits used in model creation (used to setup pruning)
        n_bits = self.module_.n_a_bits

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
            y_pred = np.zeros((X.shape[0],), np.int32)
            for idx, x in enumerate(X):
                q_x = self.quantized_module_.quantize_input(x).reshape(1, -1)
                y_pred[idx] = self.quantized_module_.forward_fhe.run(q_x).argmax(axis=1)
            return y_pred

        # For prediction in the clear we call the super class which, in turn,
        # will end up calling .infer of this class
        return super().predict(X)
