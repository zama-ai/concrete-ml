"""Scikit-learn interface for concrete quantized neural networks."""

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import brevitas.nn as qnn
import numpy
import torch
import torch.nn.utils.prune as pruning
from sklearn.base import clone
from skorch.classifier import NeuralNetClassifier as SKNeuralNetClassifier
from skorch.regressor import NeuralNetRegressor as SKNeuralNetRegressor
from torch import nn

from ..common.debugging import assert_true
from ..common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE, check_dtype_and_cast
from .base import QuantizedTorchEstimatorMixin

QNN_AUTO_KWARGS = ["module__n_outputs", "module__input_dim"]
QNN_FLOAT_DTYPE = numpy.float32
QNN_INT_DTYPE = numpy.int64


def _check_input_output_kwargs(kwargs: Dict[str, Any]) -> None:
    """Check that a QNN model is not constructed with automatically computed parameters.

    Args:
        kwargs (dict): The keyword arguments to check

    Raises:
        ValueError: if the automatically computed parameters are present in the keyword args
    """

    for kwarg in QNN_AUTO_KWARGS:
        if kwarg in kwargs:
            raise ValueError(
                f"The number of inputs and outputs of the neural network "
                "are determined automatically in .fit, based on the data-set. Setting "
                f"`{kwarg}` manually is forbidden."
            )


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
        n_accum_bits=MAX_BITWIDTH_BACKWARD_COMPATIBLE,
        n_prune_neurons_percentage=0.0,
        activation_function=nn.ReLU,
        quant_narrow=False,
        quant_signed=True,
    ):  # pylint: disable=too-many-arguments
        """Sparse Quantized Neural Network constructor.

        Args:
            input_dim: Number of dimensions of the input data
            n_layers: Number of linear layers for this network
            n_outputs: Number of output classes or regression targets
            n_w_bits: Number of weight bits
            n_a_bits: Number of activation and input bits
            n_accum_bits: Maximal allowed bitwidth of intermediate accumulators
            n_hidden_neurons_multiplier: The number of neurons on the hidden will be the number
                of dimensions of the input multiplied by `n_hidden_neurons_multiplier`. Note that
                pruning is used to adjust the accumulator size to attempt to
                keep the maximum accumulator bitwidth to
                `n_accum_bits`, meaning that not all hidden layer neurons will be active.
                The default value for `n_hidden_neurons_multiplier` is chosen for small dimensions
                of the input. Reducing this value decreases the FHE inference time considerably
                but also decreases the robustness and accuracy of model training.
            n_prune_neurons_percentage: How many neurons to prune on the hidden layers. This
                should be used mostly through the dedicated `.prune()` mechanism. This can
                be used in when setting `n_hidden_neurons_multiplier` high (3-4), once good accuracy
                is obtained, to speed up the model in FHE.
            activation_function: a torch class that is used to construct activation functions in
                the network (e.g. torch.ReLU, torch.SELU, torch.Sigmoid, etc)
            quant_narrow : whether this network should use narrow range quantized integer values
            quant_signed : whether to use signed quantized integer values

        Raises:
            ValueError: if the parameters have invalid values or the computed accumulator bitwidth
                        is zero
        """

        super().__init__()

        self.features = nn.Sequential()
        in_features = input_dim

        self.n_layers = n_layers

        if n_layers <= 0:
            raise ValueError(
                f"Invalid number of layers: {n_layers}, at least one intermediary layers is needed"
            )

        if n_w_bits <= 0 or n_a_bits <= 0:
            raise ValueError("The weight & activation quantization bitwidth cannot be less than 1")

        for idx in range(n_layers):
            out_features = (
                n_outputs if idx == n_layers - 1 else int(input_dim * n_hidden_neurons_multiplier)
            )

            quant_name = f"quant{idx}"
            quantizer = qnn.QuantIdentity(
                bit_width=n_a_bits,
                return_quant_tensor=True,
                narrow_range=quant_narrow,
                signed=quant_signed,
            )

            layer_name = f"fc{idx}"
            layer = qnn.QuantLinear(
                in_features,
                out_features,
                True,
                weight_bit_width=n_w_bits,
                bias_quant=None,
                weight_narrow_range=quant_narrow,
                narrow_range=quant_narrow,
                signed=quant_signed,
            )

            self.features.add_module(quant_name, quantizer)
            self.features.add_module(layer_name, layer)

            if idx < n_layers - 1:
                self.features.add_module(f"act{idx}", activation_function())

            in_features = out_features

        self.n_w_bits = n_w_bits
        self.n_a_bits = n_a_bits
        self.n_accum_bits = n_accum_bits

        # Store input/output dimensions to check they are correct during .fit(X,y).
        # The X passed to .fit must not have different dimensions than the one given in this
        # constructor.
        self.n_outputs = n_outputs
        self.input_dim = input_dim

        self.n_prune_neurons_percentage = n_prune_neurons_percentage

        assert_true(
            self.n_prune_neurons_percentage >= 0 and self.n_prune_neurons_percentage < 1.0,
            "Pruning percentage must be expressed as a fraction between 0 and 1. A value of "
            " zero (0) means pruning is disabled",
        )
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

        prev_layer_keep_idxs = None
        layer_idx = 0
        # Iterate over all layers that have weights (Linear ones)
        for layer in self.features:
            if not isinstance(layer, (nn.Linear, qnn.QuantLinear)):
                continue

            s = layer.weight.shape
            st = [s[0], numpy.prod(s[1:])]

            # If this is a layer that should be pruned and is currently being pruned,
            # Make the pruning permanent
            if layer in self.pruned_layers and st[1] > max_neuron_connections:
                pruning.remove(layer, "weight")
                self.pruned_layers.remove(layer)

            if self.n_prune_neurons_percentage > 0.0:
                weights = layer.weight.detach().numpy()

                # Pruning all layers except the last one, but for the last one
                # we still need to remove synapses of the previous layer's pruned neurons
                if layer_idx < self.n_layers - 1:
                    # Once pruning is disabled, the weights of some neurons become 0
                    # We need to find those neurons (columns in the weight matrix).
                    # Testing for floats equal to 0 is done using an epsilon
                    neurons_removed_idx = numpy.where(numpy.sum(numpy.abs(weights), axis=1) < 0.001)
                    idx = numpy.arange(weights.shape[0])
                    keep_idxs = numpy.setdiff1d(idx, neurons_removed_idx)
                else:
                    keep_idxs = numpy.arange(weights.shape[0])

                # Now we take the indices of the neurons kept for the previous layer
                # If this is the first layer all neurons are kept
                if prev_layer_keep_idxs is None:
                    prev_layer_keep_idxs = numpy.arange(weights.shape[1])

                # Remove the pruned neurons and the weights/synapses
                # that apply to neurons removed in the previous layer
                orig_weight = layer.weight.data.clone()
                transform_weight = orig_weight[keep_idxs]
                transform_weight = transform_weight[:, prev_layer_keep_idxs]

                # Replace the weight matrix of the current layer
                layer.weight = torch.nn.Parameter(transform_weight)

                # Eliminate the biases of the neurons that were removed in this layer
                if layer.bias is not None:
                    orig_bias = layer.bias.data.clone()
                    transform_bias = orig_bias[keep_idxs]
                    layer.bias = torch.nn.Parameter(transform_bias)

                # Save the indices of the neurons removed in this layer to
                # remove synapses in the next layer
                prev_layer_keep_idxs = keep_idxs

            layer_idx += 1

        assert_true(
            layer_idx == self.n_layers,
            "Not all layers in the network were examined as candidates for pruning",
        )

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
        layer_idx = 0
        for layer in self.features:
            if not isinstance(layer, (nn.Linear, qnn.QuantLinear)):
                continue

            s = layer.weight.shape
            st = [s[0], numpy.prod(s[1:])]

            # To satisfy accumulator bitwidth constraints each dot-product between an input line and
            # weight column must not exceed n_accum_bits bits. We thus prune the layer to have
            # at most max_neuron_connections non-zero weights
            if st[1] > max_neuron_connections and layer not in self.pruned_layers:
                pruning.l1_unstructured(layer, "weight", (st[1] - max_neuron_connections) * st[0])
                self.pruned_layers.add(layer)

            # If enabled, prune neurons for all layers except the last one,
            # which outputs the prediction
            if layer_idx < self.n_layers - 1 and self.n_prune_neurons_percentage > 0.0:
                # Use L2-norm structured pruning, using the torch ln_structured
                # function, with norm=2 and axis=0 (output/neuron axis)
                pruning.ln_structured(layer, "weight", self.n_prune_neurons_percentage, 2, 0)

            # Note this is counting only Linear layers
            layer_idx += 1

        assert_true(
            layer_idx == self.n_layers,
            "Not all layers in the network were examined as candidates for pruning",
        )

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


# pylint: disable-next=abstract-method
class QuantizedSkorchEstimatorMixin(QuantizedTorchEstimatorMixin):
    """Mixin class that adds quantization features to Skorch NN estimators."""

    # Make this class accept a generic NN and not impose our SparseQuantNeuralNetImpl
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/327
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # It basically just resets quantized_module_ by calling QuantizedSkorchEstimatorMixin's
        # initialization
        super().__init__()

        # If no parameters are passed just returns
        # Used to load the model class from json
        if len(args) == 0 and len(kwargs) == 0:
            return

        kwargs.pop("n_bits", None)

        verbose = kwargs.pop("verbose", False)
        kwargs["verbose"] = verbose

        _check_input_output_kwargs(kwargs)

        # Note that our default optimizer is Adam which was found to be more stable when pruning
        self.sklearn_model_class.__init__(
            self,
            SparseQuantNeuralNetImpl,
            *args,
            **kwargs,
        )

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
        # Disable mypy error as super().infer is expected to be reachable once the model inherits
        # from Skorch
        if not self.is_fitted:
            return super().infer(x, **fit_params)  # type: ignore[misc]

        # Get a numpy array from the tensor to do quantization
        x = x.detach().cpu().numpy()

        # Once we finished the data type checks, quantize the input and perform quantized inference
        q_x = self.quantize_input(x)
        q_out = self.quantized_module_(q_x)

        # Cast back to a tensor to keep a consistent API (tensor in, tensor out)
        return torch.tensor(self.dequantize_output(q_out))

    @property
    def base_module_to_compile(self):
        """Get the module that should be compiled to FHE. In our case this is a torch nn.Module.

        Returns:
            module (nn.Module): the instantiated torch module
        """
        # Disable mypy attribute definition error as this method is expected to be reachable
        # once the model inherits from Skorch
        self.check_is_fitted()  # type: ignore[attr-defined]

        return self.module_  # type: ignore[attr-defined]

    @property
    def n_bits_quant(self) -> int:
        """Get the number of bits used for quantization.

        This is stored by the torch.nn.module instance and thus cannot be retrieved until this
        instance is created.

        Returns:
            int: the number of bits to quantize the network
        """
        # Make sure the NeuralNet was correctly fitted
        # Here we can't use the check_model_is_fitted because n_bits is called between Skorch's
        # training and the quantized_module's initialization
        # Hence, we use Skorch's `NeuralNet.check_is_fitted` method, which raises a
        # NotInitializedError exception it the model is not fitted (by checking if the
        # `module_` attribute is found)
        # Disable mypy attribute definition error as this method is expected to be reachable
        # once the model inherits from Skorch
        self.check_is_fitted()  # type: ignore[attr-defined]

        # Similarily, ignore mypy here
        return self.module_.n_a_bits  # type: ignore[attr-defined]

    def get_sklearn_params(self, deep=True):
        """Get parameters for benchmark when cloning a skorch wrapped NN.

        We must remove all parameters related to the module. Skorch takes either a class or a
        class instance for the `module` parameter. We want to pass our trained model, a class
        instance. But for this to work, we need to remove all module related constructor params.
        If not, skorch will instantiate a new class instance of the same type as the passed module
        see skorch net.py NeuralNet::initialize_instance

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained
                subobjects that are estimators. Default to True.

        Returns:
            params (dict): parameters to create an equivalent fp32 sklearn estimator for benchmark
        """
        # Retrieve the Skorch estimator's training parameters
        # Following sklearn.base.clone's documentation, parameters are obtained with get_params()
        # and then need to be copied using copy.deepcopy before passing them to the constructor
        # Following sklearn.base.clone's documentation: "Clone does a deep copy of the model in an
        # estimator without actually copying attached data. It returns a new estimator with the
        # same parameters that has not been fitted on any data."
        # See: https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        # We therefore obtained the parameters with get_params(), copy them using copy.deepcopy
        # and then pass them to the constructor.
        # Disable mypy error as super().get_params is expected to be reachable once the model
        # inherits from Skorch
        new_object_params = super().get_params(deep=deep)  # type: ignore[misc]

        # The `module` parameter needs to be removed as we need to rebuild it separately
        # This key is only present for NeuralNetClassifiers that don't fix the module type,
        # otherwise, it may have already been removed (e.g. by FixedTypeSkorchNeuralNet)
        new_object_params.pop("module", None)

        # Retrieve all parameters related to the module
        module_params = [name for name in new_object_params if "module__" in name]

        # Remove all parameters related to the module
        for name in module_params:
            new_object_params.pop(name)

        # Deepcopy the estimator's training parameters
        for name, param in new_object_params.items():
            new_object_params[name] = copy.deepcopy(param)

        return new_object_params

    # pylint: disable-next=unused-argument
    def on_train_end(self, net, X=None, y=None, **kwargs):
        """Call back when training is finished by the skorch wrapper.

        Check if the underlying neural net has a callback for this event and, if so, call it.

        Args:
            net: estimator for which training has ended (equal to self)
            X: data
            y: targets
            kwargs: other arguments
        """
        # Disable mypy attribute definition error as this method is expected to be reachable
        # once the model inherits from Skorch
        train_end_callback = getattr(
            self.module_, "on_train_end", None  # type: ignore[attr-defined]
        )
        if callable(train_end_callback):
            train_end_callback()


class FixedTypeSkorchNeuralNet:
    """A mixin with a helpful modification to a skorch estimator that fixes the module type."""

    # The number of outputs of the underlying neural-net
    module__n_outputs: Optional[int] = None
    module__input_dim: Optional[int] = None

    def get_params(self, deep=True, **kwargs):
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
            **kwargs: any additional parameters to pass to the sklearn BaseEstimator class

        Returns:
            params : dict, Parameter names mapped to their values.
        """
        # Here, the `get_params` method is the `NeuralNet.get_params` method from Skorch
        # Disable mypy error and pylint no-member error as super().get_params is expected to be
        # reachable once the model inherits from Skorch
        # pylint: disable-next=no-member
        params = super().get_params(deep, **kwargs)  # type: ignore[misc]

        # Remove the module key since our NN skorch class imposes SparseQuantNeuralNetImpl as
        # the NN model type. Therefore, when cloning this estimator we don't need to pass
        # module again, it's passed by this class's constructor
        params.pop("module")

        # Remove the parameters that are auto-computed by `fit`
        for kwarg in QNN_AUTO_KWARGS:
            params.pop(kwarg, None)

        return params

    def prune(self, X, y, n_prune_neurons_percentage, **fit_params):
        """Prune a copy of this NeuralNetwork model.

        This can be used when the number of neurons on the hidden layers is too high. For example,
        when creating a Neural Network model with `n_hidden_neurons_multiplier` high (3-4), it
        can be used to speed up the model inference in FHE. Many times, up to 50% of
        neurons can be pruned without losing accuracy, when using this function to fine-tune
        an already trained model with good accuracy. This method should be used
        once good accuracy is obtained.

        Args:
            X : training data, can be a torch.tensor, numpy.ndarray or pandas DataFrame
            y : training targets, can be a torch.tensor, numpy.ndarray or pandas DataFrame
            n_prune_neurons_percentage : percentage of neurons to remove. A value of 0 means
                no neurons are removed and a value of 1.0 means 100% of neurons would be removed.
            fit_params: additional parameters to pass to the forward method of the underlying
                nn.Module

        Returns:
            result: a new pruned copy of this NeuralNetClassifier or NeuralNetRegressor

        Raises:
            ValueError: if the model has not been trained or if the model is one that has already
                been pruned
        """
        # Disable several mypy attribute definition error and pylint no-member error as this
        # attribute is expected to be reachable once the model inherits from Skorch
        # pylint: disable=no-member
        if not self.initialized_:  # type: ignore[attr-defined]
            raise ValueError("Can only prune another NeuralNet model that is already already fit.")

        if self.module_.n_prune_neurons_percentage > 0.0:  # type: ignore[attr-defined]
            raise ValueError(
                "Cannot apply structured pruning optimization to an already pruned model"
            )

        if n_prune_neurons_percentage >= 1.0 or n_prune_neurons_percentage < 0:
            raise ValueError(
                f"Valid values for `n_prune_neurons_percentage` are in the [0..1) range, but "
                f"{n_prune_neurons_percentage} was given."
            )

        model_copy = clone(self)

        # Copy the input/output dims, as they are kept constant. These
        # are usually computed by .fit, but here we want to manually instantiate the model
        # before .fit() in order to fine-tune the original model
        model_copy.module__input_dim = self.module__input_dim
        model_copy.module__n_outputs = self.module__n_outputs

        # Create .module_ if the .module is a class (not an instance)
        model_copy.initialize()

        # Deactivate the default pruning
        model_copy.module_.make_pruning_permanent()  # type: ignore[attr-defined]

        # Load the original model
        model_copy.module_.load_state_dict(self.module_.state_dict())  # type: ignore[attr-defined]

        # Set the new pruning amount
        model_copy.module_.n_prune_neurons_percentage = (
            n_prune_neurons_percentage  # type: ignore[attr-defined]
        )

        # Enable pruning again, this time with structured pruning
        model_copy.module_.enable_pruning()  # type: ignore[attr-defined]

        # The .module_ was initialized manually, prevent .fit from creating a new one
        model_copy.warm_start = True  # pylint: disable=attribute-defined-outside-init

        # Now, fine-tune the original module with structured pruning
        model_copy.fit(X, y, **fit_params)

        # pylint: enable=no-member

        return model_copy


# pylint: disable-next=too-many-ancestors, abstract-method
class NeuralNetClassifier(
    FixedTypeSkorchNeuralNet,
    QuantizedSkorchEstimatorMixin,
    SKNeuralNetClassifier,
):
    """Scikit-learn interface for quantized FHE compatible neural networks.

    This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator.
    It uses the skorch package to handle training and scikit-learn compatibility,
    and adds quantization and compilation functionality. The neural network implemented by this
    class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

    The datatypes that are allowed for prediction by this wrapper are more restricted than
    standard scikit-learn estimators as this class needs to predict in FHE and network inference
    executor is the NumpyModule.

    Inputs that are float64 will be casted to float32 before training as this should
    not have a significant impact on the model's performances. If the targets are integers of
    lower bitwidth, they will be safely casted to int64. Else, an error is raised.
    """

    sklearn_model_class: Callable = SKNeuralNetClassifier
    _is_a_public_cml_model = True

    # pylint: disable-next=useless-super-delegation
    def __init__(
        self,
        *args,
        criterion=torch.nn.CrossEntropyLoss,
        classes=None,
        optimizer=torch.optim.Adam,
        **kwargs,
    ):
        # Call QuantizedSkorchEstimatorMixin's __init__ method
        super().__init__(
            *args,
            criterion=criterion,
            classes=classes,
            optimizer=optimizer,
            **kwargs,
        )

    # Disable pylint here because we add an additional argument to .predict,
    # with respect to the base class .predict method.
    # pylint: disable=arguments-differ
    def predict(self, X, execute_in_fhe: bool = False) -> numpy.ndarray:
        # We just need to do argmax on the predicted probabilities
        # First we get the predicted class indices
        y_index_pred = self.predict_proba(X, execute_in_fhe=execute_in_fhe).argmax(axis=1)

        # Finally, return the class names corresponding to the indices
        return self.classes_[y_index_pred]

    # pylint: disable-next=attribute-defined-outside-init
    def fit(self, X, y, *args, **kwargs) -> Any:
        # Check that inputs are float32 and targets are int64. If inputs are float64, they will be
        # casted to float32 as this should not have a great impact on the model's performances. If
        # the targets are integers of lower bitwidth, they will be safely casted to int64. Else, an
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

        # Call QuantizedSkorchEstimatorMixin's fit method
        return super().fit(X, y, *args, **kwargs)

    def fit_benchmark(self, X, y, *args, **kwargs) -> Tuple[Any, Any]:
        # Check that inputs are float32 and targets are int64. If inputs are float64, they will be
        # casted to float32 as this should not have a great impact on the model's performances. If
        # the targets are integers of lower bitwidth, they will be safely casted to int64. Else, an
        # error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network classifier input")
        y = check_dtype_and_cast(y, "int64", error_information="Neural Network classifier target")

        # Call QuantizedSkorchEstimatorMixin's fit_benchmark method
        return super().fit_benchmark(X, y, *args, **kwargs)

    def predict_proba(self, X, execute_in_fhe: bool = False) -> numpy.ndarray:
        # Check that inputs are float32. If they are, they will be casted to float32 as this
        # should not have a great impact on the model's performances. Else, an error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network classifier input")

        # Call QuantizedSkorchEstimatorMixin's predict_proba method
        return super().predict_proba(X, execute_in_fhe)


# pylint: disable-next=too-many-ancestors, abstract-method
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

    Inputs and targets that are float64 will be casted to float32 before training as this should
    not have a significant impact on the model's performances. An error is raised if these values
    are not floating points.
    """

    sklearn_model_class: Callable = SKNeuralNetRegressor
    _is_a_public_cml_model = True

    # pylint: disable-next=useless-super-delegation
    def __init__(
        self,
        *args,
        optimizer=torch.optim.Adam,
        **kwargs,
    ):
        # Call QuantizedSkorchEstimatorMixin's __init__ method
        super().__init__(
            *args,
            optimizer=optimizer,
            **kwargs,
        )

    def fit(self, X, y, *args, **kwargs) -> Any:
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

        # Call QuantizedSkorchEstimatorMixin's fit method
        return super().fit(X, y, *args, **kwargs)

    def fit_benchmark(self, X, y, *args, **kwargs) -> Tuple[Any, Any]:
        # Check that inputs and targets are float32. If they are float64, they will be casted to
        # float32 as this should not have a great impact on the model's performances. Else, an error
        # is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network regressor input")
        y = check_dtype_and_cast(y, "float32", error_information="Neural Network regressor target")

        # Call QuantizedSkorchEstimatorMixin's fit_benchmark method
        return super().fit_benchmark(X, y, *args, **kwargs)

    def predict_proba(self, X, execute_in_fhe: bool = False) -> numpy.ndarray:
        # Check that inputs are float32. If they are float64, they will be casted to float32 as
        # this should not have a great impact on the model's performances. Else, an error is raised.
        X = check_dtype_and_cast(X, "float32", error_information="Neural Network regressor input")

        # Call QuantizedSkorchEstimatorMixin's predict_proba method
        return super().predict_proba(X, execute_in_fhe)
