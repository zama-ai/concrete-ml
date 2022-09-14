"""Scikit-learn interface for concrete quantized neural networks."""

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name

from abc import abstractmethod

import brevitas.nn as qnn
import numpy
import torch
import torch.nn.utils.prune as pruning
from concrete.numpy import MAXIMUM_TLU_BIT_WIDTH
from skorch.classifier import NeuralNetClassifier as SKNeuralNetClassifier
from skorch.regressor import NeuralNetRegressor as SKNeuralNetRegressor
from torch import nn

from .base import QuantizedTorchEstimatorMixin


class SparseQuantNeuralNetImpl(nn.Module):
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
        n_outputs,
        n_hidden_neurons_multiplier=4,
        n_w_bits=3,
        n_a_bits=3,
        n_accum_bits=MAXIMUM_TLU_BIT_WIDTH,
        activation_function=nn.ReLU,
    ):
        """Sparse Quantized Neural Network constructor.

        Args:
            input_dim: Number of dimensions of the input data
            n_layers: Number of linear layers for this network
            n_outputs: Number of output classes or regression targets
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

        if n_outputs < 1:
            raise ValueError(
                f"Invalid number of outputs: {n_outputs}, n_outputs should be larger than one"
            )

        if n_w_bits <= 0 or n_a_bits <= 0:
            raise ValueError("The weight & activation quantization bitwidth cannot be less than 1")

        for idx in range(n_layers):
            out_features = (
                n_outputs if idx == n_layers - 1 else input_dim * n_hidden_neurons_multiplier
            )

            quant_name = f"quant{idx}"
            quantizer = qnn.QuantIdentity(bit_width=n_a_bits, return_quant_tensor=True)

            layer_name = f"fc{idx}"
            layer = qnn.QuantLinear(
                in_features, out_features, True, weight_bit_width=n_w_bits, bias_quant=None
            )

            self.features.add_module(quant_name, quantizer)
            self.features.add_module(layer_name, layer)

            if idx < n_layers - 1:
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
            numpy.floor(
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
            if not isinstance(layer, nn.Linear) and not isinstance(layer, qnn.QuantLinear):
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

    def on_train_end(self):
        """Call back when training is finished, can be useful to remove training hooks."""
        self.make_pruning_permanent()


class QuantizedSkorchEstimatorMixin(QuantizedTorchEstimatorMixin):
    """Mixin class that adds quantization features to Skorch NN estimators."""

    @property
    @abstractmethod
    def base_estimator_type(self):
        """Get the sklearn estimator that should be trained by the child class."""

    def infer(self, x, **fit_params):
        """Perform a single inference step on a batch of data.

        This method is specific to Skorch estimators.

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

    @property
    def base_module_to_compile(self):
        """Get the module that should be compiled to FHE. In our case this is a torch nn.Module.

        Returns:
            module (nn.Module): the instantiated torch module
        """
        return self.module_

    @property
    def n_bits_quant(self):
        """Return the number of quantization bits.

        This is stored by the torch.nn.module instance and thus cannot be retrieved until this
        instance is created.

        Returns:
            n_bits (int): the number of bits to quantize the network

        Raises:
            ValueError: with skorch estimators, the `module_` is not instantiated until .fit() is
                called. Thus this estimator needs to be .fit() before we get the quantization number
                of bits. If it is not trained we raise an exception
        """

        module_attr = getattr(self, "module_", None)
        if module_attr is None:
            raise ValueError("NN Classifier must be trained before getting the number of bits")
        return self.module_.n_a_bits

    def get_params_for_benchmark(self):
        """Get parameters for benchmark when cloning a skorch wrapped NN.

        We must remove all parameters related to the module. Skorch takes either a class or a
        class instance for the `module` parameter. We want to pass our trained model, a class
        instance. But for this to work, we need to remove all module related constructor params.
        If not, skorch will instantiate a new class instance of the same type as the passed module
        see skorch net.py NeuralNet::initialize_instance

        Returns:
            params (dict): parameters to create an equivalent fp32 sklearn estimator for benchmark
        """
        new_object_params = super().get_params_for_benchmark()
        module_params = [name for name in new_object_params.keys() if "module__" in name]
        for name in module_params:
            new_object_params.pop(name)
        return new_object_params

    # pylint: disable=unused-argument
    def on_train_end(self, net, X=None, y=None, **kwargs):
        """Call back when training is finished by the skorch wrapper.

        Check if the underlying neural net has a callback for this event and, if so, call it.

        Args:
            net: estimator for which training has ended (equal to self)
            X: data
            y: targets
            kwargs: other arguments
        """
        train_end_callback = getattr(self.module_, "on_train_end", None)
        if callable(train_end_callback):
            train_end_callback()


class FixedTypeSkorchNeuralNet:
    """A mixin with a helpful modification to a skorch estimator that fixes the module type."""

    def get_params(self, deep=True, **kwargs):
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
            **kwargs: any additional parameters to pass to the sklearn BaseEstimator class

        Returns:
            params : dict, Parameter names mapped to their values.
        """
        # Known but with pylint when mixin is mixed with a class in a different file
        # pylint: disable=no-member
        params = super().get_params(deep, **kwargs)
        # pylint: enable=no-member

        # Remove the module key since our NN skorch class imposes SparseQuantNeuralNetImpl as
        # the NN model type. Therefore, when cloning this estimator we don't need to pass
        # module again, it's passed by this class's constructor
        params.pop("module")
        return params


class NeuralNetClassifier(
    FixedTypeSkorchNeuralNet, QuantizedSkorchEstimatorMixin, SKNeuralNetClassifier
):
    """Scikit-learn interface for quantized FHE compatible neural networks.

    This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator.
    It uses the skorch package to handle training and scikit-learn compatibility,
    and adds quantization and compilation functionality. The neural network implemented by this
    class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

    The datatypes that are allowed for prediction by this wrapper are more restricted than
    standard scikit-learn estimators as this class needs to predict in FHE and network inference
    executor is the NumpyModule.
    """

    # FIXME: make this class accept a generic NN and not impose our SparseQuantNeuralNetImpl
    # see https://github.com/zama-ai/concrete-ml-internal/issues/327
    def __init__(
        self,
        *args,
        criterion=torch.nn.CrossEntropyLoss,
        classes=None,
        optimizer=torch.optim.Adam,
        **kwargs,
    ):
        # It basically just sets quantized_module_ and _onnx_model_ to None
        # It calls the init of QuantizedSkorchEstimatorMixin
        super().__init__()

        # If no parameters are passed just returns
        # Used to load the model class from json
        if len(args) == 0 and len(kwargs) == 0:
            return
        # A helper for users so they don't need to import torch directly
        args_to_convert_to_tensor = ["criterion__weight"]
        for arg_name in args_to_convert_to_tensor:
            if arg_name in kwargs and isinstance(kwargs[arg_name], numpy.ndarray):
                kwargs[arg_name] = torch.from_numpy(kwargs[arg_name]).float()

        n_classes = kwargs["module__n_outputs"]
        if n_classes < 2:
            raise ValueError(
                f"Invalid number of classes: {str(n_classes)}, "
                "n_outputs should be larger than one"
            )

        kwargs.pop("n_bits", None)

        # Note that our default optimizer is Adam which was found to be more stable when pruning
        SKNeuralNetClassifier.__init__(
            self,
            SparseQuantNeuralNetImpl,
            *args,
            criterion=criterion,
            classes=classes,
            optimizer=optimizer,
            **kwargs,
        )

    @property
    def base_estimator_type(self):
        return SKNeuralNetClassifier

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

        # We just need to do argmax on the predicted probabilities
        return self.predict_proba(X, execute_in_fhe=execute_in_fhe).argmax(axis=1)

    def fit(self, X, y, **fit_params):
        # We probably can't handle all cases since per Skorch documentation they handle:
        #  * numpy arrays
        #  * torch tensors
        #  * pandas DataFrame or Series
        #  * scipy sparse CSR matrices
        #  * a dictionary of the former three
        #  * a list/tuple of the former three
        #  * a Dataset
        # which is a bit much since they don't necessarily have the same interfaces to handle types
        if isinstance(X, numpy.ndarray) and (X.dtype != numpy.float32):
            X = X.astype(numpy.float32)
        if isinstance(y, numpy.ndarray) and (y.dtype != numpy.int64):
            y = y.astype(numpy.int64)
        return super().fit(X, y, **fit_params)


class NeuralNetRegressor(
    FixedTypeSkorchNeuralNet, QuantizedSkorchEstimatorMixin, SKNeuralNetRegressor
):
    """Scikit-learn interface for quantized FHE compatible neural networks.

    This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator.
    It uses the skorch package to handle training and scikit-learn compatibility,
    and adds quantization and compilation functionality. The neural network implemented by this
    class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

    The datatypes that are allowed for prediction by this wrapper are more restricted than
    standard scikit-learn estimators as this class needs to predict in FHE and network inference
    executor is the NumpyModule.
    """

    # FIXME: make this class accept a generic NN and not impose our SparseQuantNeuralNetImpl
    # see https://github.com/zama-ai/concrete-ml-internal/issues/327
    def __init__(
        self,
        *args,
        optimizer=torch.optim.Adam,
        **kwargs,
    ):
        super().__init__()
        # If no parameters are passed just return
        # Used to load the model class from json
        if len(args) == 0 and len(kwargs) == 0:
            return

        kwargs.pop("n_bits", None)

        # Note that our default optimizer is Adam which was found to be more stable when pruning
        SKNeuralNetRegressor.__init__(
            self,
            SparseQuantNeuralNetImpl,
            *args,
            optimizer=optimizer,
            **kwargs,
        )

    @property
    def base_estimator_type(self):
        return SKNeuralNetRegressor

    def fit(self, X, y, **fit_params):
        # We probably can't handle all cases since per Skorch documentation they handle:
        #  * numpy arrays
        #  * torch tensors
        #  * pandas DataFrame or Series
        #  * scipy sparse CSR matrices
        #  * a dictionary of the former three
        #  * a list/tuple of the former three
        #  * a Dataset
        # which is a bit much since they don't necessarily have the same interfaces to handle types
        if isinstance(X, numpy.ndarray) and (X.dtype != numpy.float32):
            X = X.astype(numpy.float32)
        if isinstance(y, numpy.ndarray) and (y.dtype != numpy.float32):
            y = y.astype(numpy.float32)
        return super().fit(X, y, **fit_params)
