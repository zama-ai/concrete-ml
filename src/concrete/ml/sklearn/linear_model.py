"""Implement sklearn linear model."""

import itertools
import time
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy
import sklearn.linear_model
from concrete.fhe import Configuration
from concrete.fhe import Value as EncryptedValue
from sklearn.preprocessing import LabelEncoder

from ..common.utils import FheMode
from ..onnx.ops_impl import numpy_sigmoid
from ..quantization import QuantizedModule
from ..torch.compile import _compile_torch_or_onnx_model
from ._fhe_training_utils import LogisticRegressionTraining, binary_cross_entropy
from .base import (
    Data,
    SklearnLinearClassifierMixin,
    SklearnLinearRegressorMixin,
    SklearnSGDClassifierMixin,
    SklearnSGDRegressorMixin,
    Target,
)


# pylint: disable=invalid-name,too-many-instance-attributes,too-many-lines
class LinearRegression(SklearnLinearRegressorMixin):
    """A linear regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LinearRegression please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    sklearn_model_class = sklearn.linear_model.LinearRegression

    _is_a_public_cml_model = True

    def __init__(
        self,
        n_bits=8,
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["fit_intercept"] = self.fit_intercept
        metadata["copy_X"] = self.copy_X
        metadata["n_jobs"] = self.n_jobs
        metadata["positive"] = self.positive

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # scikit-learn
        obj.fit_intercept = metadata["fit_intercept"]
        obj.copy_X = metadata["copy_X"]
        obj.n_jobs = metadata["n_jobs"]
        obj.positive = metadata["positive"]
        return obj


# pylint: disable-next=too-many-ancestors
class SGDClassifier(SklearnSGDClassifierMixin):
    """An FHE linear classifier model fitted with stochastic gradient descent.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.
        fit_encrypted (bool): Indicate if the model should be fitted in FHE or not. Default to
            False.
        parameters_range (Optional[Tuple[float, float]]): Range of values to consider for the
            model's parameters when compiling it after training it in FHE (if fit_encrypted is set
            to True). Default to None.
        batch_size (int): Batch size to consider for the gradient descent during FHE training (if
            fit_encrypted is set to True). Default to 8.

    For more details on SGDClassifier please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    """

    sklearn_model_class = sklearn.linear_model.SGDClassifier

    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments,too-many-locals
    def __init__(
        self,
        n_bits=8,
        fit_encrypted=False,
        parameters_range=None,
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter: int = 1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        # Concrete ML attributes for FHE training
        # These values are hardcoded for now
        # We don't expose them in the __init__ arguments but they are taken
        # into account when training, so we can just modify them manually.
        # The number of bits used for training should be adjusted according to n-bits
        # but for now we use this hardcoded values.
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4205
        self.n_bits_training = 6
        self.rounding_training = 7
        self.learning_rate_value = 1.0
        self.batch_size = 8
        self.training_p_error = 0.01
        self.training_fhe_configuration = None

        self.fit_encrypted = fit_encrypted
        self.parameters_range = parameters_range

        #: The random number generator to use during compilation after FHE training (if enabled)
        self.random_number_generator = numpy.random.default_rng(random_state)

        #: The quantized module used for FHE training (if enabled)
        self.training_quantized_module: Optional[QuantizedModule] = None

        #: The weight values used for FHE training (if enabled)
        self._weights_encrypted_fit: Optional[numpy.ndarray] = None

        #: The bias values used for FHE training (if enabled)
        self._bias_encrypted_fit: Optional[numpy.ndarray] = None

        # scikit-learn's attributes
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

        # Checks the coherence of some attributes
        assert (
            isinstance(self.max_iter, int) and self.max_iter >= 1
        ), "Parameter 'max_iter' must be an integer in range [1, inf]"
        assert isinstance(self.tol, (float, type(None)))

        # Checks and warnings for FHE training
        if self.fit_encrypted:
            self.classes_: Optional[numpy.ndarray] = None

            # Check the presence of mandatory attributes
            if self.loss != "log_loss":
                raise ValueError(
                    f"Only 'log_loss' is currently supported if FHE "
                    f"training is enabled ({fit_encrypted=}). Got {loss=}"
                )

            if self.parameters_range is None:
                raise ValueError(
                    "Setting 'parameters_range' is mandatory if FHE training is enabled "
                    f"({fit_encrypted=}). Got {parameters_range=}"
                )

        else:
            supported_losses = ["log_loss", "modified_huber"]
            if self.loss not in supported_losses:
                raise NotImplementedError(
                    f"Only one of {supported_losses} loss is supported. Got {self.loss}."
                )

    def get_sklearn_params(self, deep: bool = True) -> dict:
        # Here, the `get_params` method is the `BaseEstimator.get_params` method from scikit-learn
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        params = super().get_params(deep=deep)  # type: ignore[misc]

        # Remove the parameters added by Concrete ML
        params.pop("n_bits", None)
        params.pop("n_bits_training", None)
        params.pop("rounding_training", None)
        params.pop("fit_encrypted", None)
        params.pop("parameters_range", None)
        params.pop("batch_size", None)
        params.pop("learning_rate_value", None)

        return params

    def _get_training_quantized_module(
        self,
        x_min: numpy.ndarray,
        x_max: numpy.ndarray,
        device: str,
    ) -> QuantizedModule:
        """Get the quantized module for FHE training.

        This method builds the quantized module and fhe-circuit needed to train the model in FHE.

        Args:
            x_min (numpy.ndarray): The minimum value to consider for each feature over the samples.
            x_max (numpy.ndarray): The maximum value to consider for each feature over the samples.
            device (str): FHE compilation device, can be either 'cpu' or 'cuda'.

        Returns:
            (QuantizedModule): The quantized module containing the FHE circuit for training.

        """
        # Mypy
        assert self.parameters_range is not None

        # Compile and return the training quantized module
        # 54 = 2 classes * 3 values for x * 2 values for the weights * 2 values for the bias
        # Number of combination of extreme values
        combinations = list(
            itertools.product(
                [1.0, 0.0],  # Labels
                [x_min, x_max, numpy.zeros(x_min.shape)],  # Data-range
                [self.parameters_range[0], self.parameters_range[1]],  # Weights
                [self.parameters_range[0], self.parameters_range[1]],  # Bias
            )
        )

        compile_size = len(combinations)
        n_targets = 1

        # Generate the input values to consider for compilation
        x_compile_set = numpy.empty((compile_size, self.batch_size, x_min.shape[0]))

        # Generate the target values to consider for compilation
        # Update this once we support multi-class
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4182
        y_compile_set = numpy.empty((compile_size, self.batch_size, n_targets))

        # Generate the weight values to consider for compilation
        weights_compile_set = numpy.empty((compile_size, x_min.shape[0], n_targets))

        # Generate the bias values to consider for compilation
        bias_compile_set = numpy.empty((compile_size, 1, n_targets))

        compile_set = (x_compile_set, y_compile_set, weights_compile_set, bias_compile_set)

        # Bound values are hard-coded in order to make sure that the circuit never overflows
        for index, (label, x_value, coef_value, bias_value) in enumerate(combinations):
            compile_set[0][index] = x_value
            compile_set[1][index] = label
            compile_set[2][index] = coef_value

            if not self.fit_intercept:
                bias_value *= 0.0

            compile_set[3][index] = bias_value

        # Instantiate the LogisticRegressor model
        trainer = LogisticRegressionTraining(
            learning_rate=self.learning_rate_value,
            iterations=1,
            fit_bias=self.fit_intercept,
        )

        if self.training_fhe_configuration is None:
            configuration = Configuration()
        else:
            configuration = self.training_fhe_configuration

        # Enable the underlying FHE circuit to be composed with itself
        # This feature is used in order to be able to iterate in the clear n times without having
        # to encrypt/decrypt the weight/bias values between each loop
        configuration.composable = True

        composition_mapping = {0: 2, 1: 3}

        # Compile the model using the compile set
        if self.verbose:
            print(f"Compiling training circuit on device '{device}'...")

        start = time.time()
        training_quantized_module = _compile_torch_or_onnx_model(
            trainer,
            compile_set,
            n_bits=self.n_bits_training,
            rounding_threshold_bits=self.rounding_training,
            p_error=self.training_p_error,
            configuration=configuration,
            reduce_sum_copy=True,
            composition_mapping=composition_mapping,
            device=device,
        )
        end = time.time()

        if self.verbose:
            print(f"Compilation took {end - start:.4f} seconds.")

        return training_quantized_module

    def _decrypt_dequantize_training_output(
        self,
        weights_enc: Union[numpy.ndarray, EncryptedValue],
        bias_enc: Union[numpy.ndarray, EncryptedValue],
        fhe: Union[str, FheMode] = FheMode.DISABLE,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Decrypt and de-quantize the outputs using the training circuit.

        Args:
            weights_enc (Union[numpy.ndarray, EncryptedValue]): The weight values to decrypt (if
                encrypted) and de-quantize.
            bias_enc (Union[numpy.ndarray, EncryptedValue]): The bias values to decrypt (if
                encrypted) and de-quantize.
            fhe (Union[str, FheMode]): The mode to use for FHE training.
                Can be FheMode.DISABLE for Concrete ML Python (quantized) training,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values. Default to
                FheMode.DISABLE.

        Returns:
            weights_float, bias_float (Tuple[numpy.ndarray, numpy.ndarray]): The weight and bias
                float values.
        """
        # Mypy
        assert self.training_quantized_module is not None
        assert self.training_quantized_module.fhe_circuit is not None

        # If the training is done in FHE, decrypt the weight and bias values
        if fhe == "execute":
            q_weights, q_bias = self.training_quantized_module.fhe_circuit.decrypt(
                weights_enc, bias_enc
            )

        else:
            q_weights, q_bias = weights_enc, bias_enc

        weights_float, bias_float = self.training_quantized_module.dequantize_output(
            q_weights, q_bias
        )

        # Reshape parameters to fit what scikit-learn expects
        weights_float, bias_float = weights_float.squeeze(0), bias_float.squeeze(0)

        return weights_float, bias_float

    # pylint: disable-next=too-many-branches, too-many-statements, too-many-locals
    def _fit_encrypted(
        self,
        X,
        y,
        fhe: Union[str, FheMode] = FheMode.DISABLE,
        coef_init: Optional[numpy.ndarray] = None,
        intercept_init: Optional[numpy.ndarray] = None,
        is_partial_fit: bool = False,
        classes: Optional[numpy.ndarray] = None,
        device: str = "cpu",
    ):
        """Fit SGDClassifier in FHE.

        The is the underlying function that fits the model in FHE if 'fit_encrypted' is enabled.
        A quantized module is first built in order to generate the FHE circuit need for training.
        Then, the method iterates over it in the clear so that outputs of an iteration are used as
        inputs for the following iteration. Thanks to Concrete's composition feature, no
        encryption/decryption steps are needed when the training is executed in FHE.

        For more details on some of these arguments please refer to:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
                It mush have a shape of (n_samples, n_features).
            y (Target): The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            fhe (Union[str, FheMode]): The mode to use for FHE training.
                Can be FheMode.DISABLE for Concrete ML Python (quantized) training,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values. Default to
                FheMode.DISABLE.
            coef_init (Optional[numpy.ndarray]): The initial coefficients to warm-start the
                optimization. Default to None.
            intercept_init (Optional[numpy.ndarray]): The initial intercept to warm-start the
                optimization. Default to None.
            is_partial_fit (bool): Indicates if this fit represents a partial fit. A partial fit is
                similar to a fit but with only a single iteration.
            classes (Optional[numpy.ndarray]): should be specified in the first call to partial fit.
            device: FHE compilation device, can be either 'cpu' or 'cuda'.

        Returns:
            The fitted estimator.

        Raises:
            NotImplementedError: If the target values are not binary and 2D, or in the target values
                are not 1D.
            ValueError: If called from `partial_fit`, and classes is None on first call.
        """
        if len(X.shape) != 2:
            raise NotImplementedError(
                "Input values must be 2D, with a shape of (n_samples, n_features), when FHE "
                f"training is enabled. Got {X.shape}"
            )

        if len(y.shape) != 1:
            raise NotImplementedError(
                "Target values must be 1D, with a shape of (n_samples,), when FHE training is "
                f"enabled. Got {y.shape}"
            )

        if classes is not None and self.classes_ is not None:
            if len(numpy.setxor1d(classes, self.classes_)) > 0:
                raise ValueError(
                    f"{classes=} is not the same as on last call to partial_fit,"
                    f" was: {self.classes_}"
                )

        n_samples, n_features = X.shape
        weight_shape = (1, n_features, 1)
        bias_shape = (1, 1, 1)

        # Build the quantized module
        # In case of a partial fit, only do so if it has not been done already (which indicates
        # that this is the partial fit's first call)
        if (not is_partial_fit) or (self.training_quantized_module is None):
            # Update this once we support multi-class
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4182
            # We need to define this here and not in the init otherwise this breaks
            # because scikit-learn assumes that as soon as the attribute exists
            # the model is fitted
            # pylint: disable=attribute-defined-outside-init
            if is_partial_fit and self.training_quantized_module is None and classes is None:
                raise ValueError("classes must be passed on the first call to partial_fit.")

            if classes is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(y)
                self.classes_ = numpy.array(self.label_encoder.classes_)
            else:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(classes)
                self.classes_ = numpy.array(classes)
            # If partial fit we should probably support adding classes along the way

            assert isinstance(self.classes_, numpy.ndarray)

            # Allow the training set to only provide a single class. This can happen, for example,
            # when running 'partial_fit' on a small batch of values. Even with a single class, the
            # model remains binary
            if len(self.classes_) != 2:
                raise NotImplementedError(
                    f"Only binary classification is currently supported when FHE training is "
                    f"enabled. Got {len(self.classes_)} labels: {self.classes_}."
                )

            # Get the inputs' extreme values
            x_min, x_max = X.min(axis=0), X.max(axis=0)

            # Build and compile the training quantized module
            self.training_quantized_module = self._get_training_quantized_module(
                x_min=x_min, x_max=x_max, device=device
            )

        y = self.label_encoder.transform(y)

        # Mypy
        assert self.training_quantized_module.fhe_circuit is not None

        # Key generation
        if fhe == "execute":  # pragma: no cover

            # Generate the keys only if necessary. This is already done using the `force=False`
            # parameter, but here we also avoid printing too much verbose if activated
            if not self.training_quantized_module.fhe_circuit.keys.are_generated:
                if self.verbose:
                    print("Key Generation...")

                start = time.time()
                self.training_quantized_module.fhe_circuit.keygen(force=False)
                end = time.time()

                if self.verbose:
                    print(f"Key generation took {end - start:.4f} seconds.")

        # Mypy
        assert self.parameters_range is not None

        # Initialize the weight values with the given ones if some are provided
        if coef_init is not None:
            weights = coef_init.reshape(weight_shape)

        # Else, if warm start is activated or this is a partial fit, use some already computed
        # weight values have if there are some
        elif (self.warm_start or is_partial_fit) and self._weights_encrypted_fit is not None:
            weights = self._weights_encrypted_fit.reshape(weight_shape)

        # Else, initialize the values randomly
        else:
            weights = self.random_number_generator.uniform(
                low=self.parameters_range[0],
                high=self.parameters_range[1],
                size=weight_shape,
            )

        # If the mode should fit the bias values as well
        if self.fit_intercept:

            # Initialize the bias values with the given ones if some are provided
            if intercept_init is not None:
                bias = intercept_init.reshape(bias_shape)

            # Else, if warm start is activated or this is a partial fit, use some already computed
            # bias values have if there are some
            elif (self.warm_start or is_partial_fit) and self._bias_encrypted_fit is not None:
                bias = self._bias_encrypted_fit.reshape(bias_shape)

            # Else, initialize the values randomly
            else:
                bias = self.random_number_generator.uniform(
                    low=self.parameters_range[0],
                    high=self.parameters_range[1],
                    size=bias_shape,
                )

        # Else, initialize the bias with zeros
        else:
            bias = numpy.zeros(bias_shape)

        # Only print this verbose once if in training using `partial_fit``
        if self.verbose and (not is_partial_fit or self.training_quantized_module is None):
            mode_string = " (simulation)" if fhe == "simulate" else ""
            print(f"Training on encrypted data{mode_string}...")

        # A partial fit is similar to running a fit with a single iteration
        max_iter = 1 if is_partial_fit else self.max_iter

        # Iterate on the batches in order to quantize and encrypt them
        X_batches_enc, y_batches_enc = [], []
        for _ in range(max_iter):

            # Sample the batches from X and y in the clear
            batch_indexes = self.random_number_generator.choice(
                n_samples, size=self.batch_size, replace=False
            )

            # Mypy
            assert isinstance(batch_indexes, numpy.ndarray)

            # Build the batches
            X_batch = X[batch_indexes].astype(float).reshape((1, self.batch_size, n_features))
            y_batch = y[batch_indexes].reshape((1, self.batch_size, 1)).astype(float)

            # The underlying quantized module expects (X, y, weight, bias) as inputs. We thus only
            # quantize the input and target values using the first and second positional parameter
            q_X_batch, q_y_batch, _, _ = self.training_quantized_module.quantize_input(
                X_batch, y_batch, None, None
            )

            # If the training is done in FHE, encrypt the input and target values
            if fhe == "execute":

                # Similarly, the underlying FHE circuit expects (X, y, weight, bias) as inputs, and
                # so does the encrypt method
                X_batch_enc, y_batch_enc, _, _ = self.training_quantized_module.fhe_circuit.encrypt(
                    q_X_batch, q_y_batch, None, None
                )

            else:
                X_batch_enc, y_batch_enc = q_X_batch, q_y_batch

            X_batches_enc.append(X_batch_enc)
            y_batches_enc.append(y_batch_enc)

        # Similarly, we only quantize the weight and bias values using the third and fourth
        # position parameter
        _, _, q_weights, q_bias = self.training_quantized_module.quantize_input(
            None, None, weights, bias
        )

        # If the training is done in FHE, encrypt the weight and bias values
        if fhe == "execute":

            # Similarly, we only encrypt using the third and fourth position parameter
            _, _, weights_enc, bias_enc = self.training_quantized_module.fhe_circuit.encrypt(
                None, None, q_weights, q_bias
            )

        else:
            weights_enc, bias_enc = q_weights, q_bias

        # This variable is used for computing the loss and handle early stopping (see at the end of
        # the loop)
        loss_value_moving_average = None

        # Iterate on the training quantized module in the clear
        for iteration_step in range(max_iter):
            X_batch_enc_i, y_batch_enc_i = (
                X_batches_enc[iteration_step],
                y_batches_enc[iteration_step],
            )
            # Train the model over one iteration
            inference_start = time.time()

            # If the training is done in FHE, execute the underlying FHE circuit directly on the
            # encrypted values
            if fhe == "execute":
                weights_enc, bias_enc = self.training_quantized_module.fhe_circuit.run(
                    X_batch_enc_i,
                    y_batch_enc_i,
                    weights_enc,
                    bias_enc,
                )

            # Else, use the quantized module on the quantized values (works for both quantized
            # clear and FHE simulation modes). It is important to note that 'quantized_forward'
            # with 'fhe="execute"' is executing Concrete's 'encrypt_run_decrypt' method, as opposed
            # to the 'run' method right above. We thus need to separate these cases since values
            # are already encrypted here.
            else:
                weights_enc, bias_enc = self.training_quantized_module.quantized_forward(
                    X_batch_enc_i, y_batch_enc_i, weights_enc, bias_enc, fhe=fhe
                )

            if self.verbose:
                print(
                    f"Iteration {iteration_step} took {time.time() - inference_start:.2f} seconds."
                )

            # If early stopping is enabled, decrypt (if needed) and de-quantize the weight and bias
            # values. Then, compute the loss and stop the training if it gets under the given
            # tolerance
            # Additionally, there is no point in computing the following in case of a partial fit,
            # as it only represents a single iteration
            if self.early_stopping and not is_partial_fit:
                weights_float, bias_float = self._decrypt_dequantize_training_output(
                    weights_enc, bias_enc, fhe=fhe
                )

                # Evaluate the model on the full dataset and compute the loss
                logits = ((X @ weights_float) + bias_float).squeeze()
                loss_value = binary_cross_entropy(y_true=y, logits=logits)

                # If this is the first training iteration, store the loss value computed above
                if loss_value_moving_average is None:
                    loss_value_moving_average = loss_value

                # Else, update the value
                else:
                    previous_loss_value_moving_average = loss_value_moving_average
                    loss_value_moving_average = (loss_value_moving_average + loss_value) / 2

                    loss_difference = numpy.abs(
                        previous_loss_value_moving_average - loss_value_moving_average
                    )

                    # If the loss gets under the given tolerance, stop the training
                    if loss_difference < self.tol:
                        break

        # Decrypt (if needed) and de-quantize the fitted weight and bias values
        fitted_weights, fitted_bias = self._decrypt_dequantize_training_output(
            weights_enc, bias_enc, fhe=fhe
        )

        # Initialize the underlying scikit-learn model if it has not already been done
        # This model should be directly initialized in the model's __init__ method instead
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3373
        if self.sklearn_model is None:

            # Retrieve the init parameters
            params = self.get_sklearn_params()

            self.sklearn_model = self.sklearn_model_class(**params)

        # Build the underlying scikit-learn model with the computed weight and bias values
        self.sklearn_model.coef_ = fitted_weights.T
        self.sklearn_model.intercept_ = fitted_bias

        # Copy over the classes since they are needed by the HB ONNX export
        self.sklearn_model.classes_ = self.classes_

        # Update the model's Concrete ML parameters
        self._weights_encrypted_fit = fitted_weights
        self._bias_encrypted_fit = fitted_bias
        self._is_fitted = True
        self._quantize_model(X)

        return self

    # The fit method's signature differs from the BaseEstimator's one for two main reasons:
    # - a new 'fhe' parameter is added in order to be able to fit the model in FHE, which is only
    #   enabled for the SGDClassifier class
    # - additional keyword arguments are exposed to make this method better match scikit-learn's
    #   fit signature
    # pylint: disable-next=arguments-differ
    def fit(  # type: ignore[override]
        self,
        X: Data,
        y: Target,
        fhe: Optional[Union[str, FheMode]] = None,
        coef_init: Optional[numpy.ndarray] = None,
        intercept_init: Optional[numpy.ndarray] = None,
        sample_weight: Optional[numpy.ndarray] = None,
        device: str = "cpu",
    ):
        """Fit SGDClassifier.

        For more details on some of these arguments please refer to:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        Training with encrypted data differs a bit from what is done by scikit-learn
        on multiple points:
        - The learning rate used is constant (self.learning_rate_value)
        - There is a batch size, it does not use the full dataset (self.batch_size)

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            fhe (Optional[Union[str, FheMode]]): The mode to use for FHE training.
                Can be FheMode.DISABLE for Concrete ML Python (quantized) training,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values. If None, training is
                done in floating points in the clear through scikit-learn. Default to None.
            coef_init (Optional[numpy.ndarray]): The initial coefficients to warm-start the
                optimization. Default to None.
            intercept_init (Optional[numpy.ndarray]): The initial intercept to warm-start the
                optimization. Default to None.
            sample_weight (Optional[numpy.ndarray]): Weights applied to individual samples (1. for
                unweighted). It is currently not supported for FHE training. Default to None.
            device: FHE compilation device, can be either 'cpu' or 'cuda'.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: if `fhe` is provided but `fit_encrypted==False`
            NotImplementedError: If parameter a 'sample_weight' is given while FHE training is
                enabled.
        """

        # If the model should be trained using FHE training
        if self.fit_encrypted:
            if fhe is None:
                fhe = "disable"
                warnings.warn(
                    "Parameter 'fhe' isn't set while FHE training is enabled.\n"
                    f"Defaulting to '{fhe=}'",
                    stacklevel=2,
                )

            # Make sure the `fhe` parameter is correct
            assert FheMode.is_valid(fhe), (
                "`fhe` mode is not supported. Expected one of 'disable' (resp. FheMode.DISABLE), "
                "'simulate' (resp. FheMode.SIMULATE) or 'execute' (resp. FheMode.EXECUTE). Got "
                f"{fhe}",
            )

            if sample_weight is not None:
                raise NotImplementedError(
                    "Parameter 'sample_weight' is currently not supported for FHE training."
                )

            return self._fit_encrypted(
                X=X,
                y=y,
                fhe=fhe,
                coef_init=coef_init,
                intercept_init=intercept_init,
                device=device,
            )

        if fhe is not None:
            raise ValueError(
                "Parameter 'fhe' should not be set when FHE training is disabled. Either set it to "
                "None for floating point training in the clear or set 'fit_encrypted' to True when "
                f"initializing the model. Got {fhe}."
            )

        # Else, train the model in floating points in the clear through scikit-learn
        return super().fit(
            X,
            y,
            coef_init=coef_init,
            intercept_init=intercept_init,
            sample_weight=sample_weight,
        )

    def partial_fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        fhe: Optional[Union[str, FheMode]] = None,
        classes=None,
    ):
        """Fit SGDClassifier for a single iteration.

        This function does one iteration of SGD training. Looping n_times over this function is
        equivalent to calling 'fit' with max_iter=n_times.

        Args:
            X (Data): The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
            y (Target): The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas
                Series or List.
            fhe (Optional[Union[str, FheMode]]): The mode to use for FHE training.
                Can be FheMode.DISABLE for Concrete ML Python (quantized) training,
                FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.
                Can also be the string representation of any of these values. If None, training is
                done in floating points in the clear through scikit-learn. Default to None.
            classes (Optional[numpy.ndarray]): The classes in the dataset.
                It needs to be provided in the first call to `partial_fit`.
                If provided in following calls it should match the classes
                provided in the first call

        Raises:
            NotImplementedError: If FHE training is disabled.
        """
        # A partial fit is similar to a fit with a single iteration. The slight differences between
        # both are handled in the encrypted method when setting `is_partial_fit` to True.
        if self.fit_encrypted:
            if fhe is None:
                fhe = "disable"
                warnings.warn(
                    "Parameter 'fhe' isn't set while FHE training is enabled.\n"
                    f"Defaulting to '{fhe=}'",
                    stacklevel=2,
                )

            # Make sure the `fhe` parameter is correct
            assert FheMode.is_valid(fhe), (
                "`fhe` mode is not supported. Expected one of 'disable' (resp. FheMode.DISABLE), "
                "'simulate' (resp. FheMode.SIMULATE) or 'execute' (resp. FheMode.EXECUTE). Got "
                f"{fhe}",
            )

            self._fit_encrypted(X=X, y=y, fhe=fhe, is_partial_fit=True, classes=classes)

        else:
            # Expose and implement partial_fit for clear training
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4184
            raise NotImplementedError("Partial fit is not currently supported for clear training.")

    def post_processing(self, y_preds: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the de-quantized predictions.

        This is called at the end of the `predict_proba` method and is only available for log loss
        and modified Huber losses. Multiclass probability estimates are derived from binary
        (one-vs.-rest) estimates by simple normalization, as recommended by Zadrozny and Elkan.

        Binary probability estimates for loss="modified_huber" are given by
        (clip(decision_function(X), -1, 1) + 1) / 2. For other loss functions
        it is necessary to perform proper probability calibration by wrapping
        the classifier with `sklearn.calibration.CalibratedClassifierCV` instead.

        Args:
            y_preds (Data): The de-quantized predictions to post-process. It mush have a shape of
                (n_samples, n_features).

        Returns:
            numpy.ndarray: The post-processed predictions, with shape (n_samples, n_classes).

        Raises:
            NotImplementedError: If the given loss is not supported.

        References:
            Zadrozny and Elkan, "Transforming classifier scores into multiclass
            probability estimates", SIGKDD'02,
            https://dl.acm.org/doi/pdf/10.1145/775047.775151

            The justification for the formula in the loss="modified_huber"
            case is in the appendix B in:
            http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
        """
        # The following lines are taken directly from scikit-learn's source code
        if self.loss == "log_loss":
            y_preds = numpy_sigmoid(y_preds)[0]

            assert isinstance(y_preds, numpy.ndarray)

            if y_preds.ndim == 1 or y_preds.shape[1] == 1:
                y_preds = y_preds.flatten()
                return numpy.vstack([1 - y_preds, y_preds]).T

            # OvR normalization, like LibLinear's predict_probability
            prob = y_preds / y_preds.sum(axis=1).reshape((y_preds.shape[0], -1))

        # The following lines are taken directly from scikit-learn's source code
        elif self.loss == "modified_huber":
            assert isinstance(self.classes_, numpy.ndarray)
            binary = len(self.classes_) == 2

            prob2 = numpy.empty(tuple())
            if binary:
                prob2 = numpy.ones((y_preds.shape[0], 2))
                prob = prob2[:, 1]

            else:
                prob = y_preds

            numpy.clip(y_preds, -1, 1, prob)
            prob += 1.0
            prob /= 2.0

            if binary:
                prob2[:, 0] -= prob
                prob = prob2

            else:
                # the above might assign zero to all classes, which doesn't
                # normalize neatly; work around this to produce uniform
                # probabilities
                prob_sum = prob.sum(axis=1)
                all_zero = prob_sum == 0

                if numpy.any(all_zero):  # pragma: no cover
                    prob[all_zero, :] = 1
                    prob_sum[all_zero] = len(self.classes_)

                # normalize
                prob /= prob_sum.reshape((prob.shape[0], -1))

        else:  # pragma: no cover
            supported_losses = ["log_loss", "modified_huber"]
            raise NotImplementedError(
                f"Only one of {supported_losses} loss is supported. Got {self.loss}."
            )

        return prob

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML for training in FHE
        metadata["n_bits"] = self.n_bits
        metadata["n_bits_training"] = self.n_bits_training
        metadata["rounding_training"] = self.rounding_training
        metadata["fit_encrypted"] = self.fit_encrypted
        metadata["parameters_range"] = self.parameters_range
        metadata["batch_size"] = self.batch_size
        metadata["learning_rate_value"] = self.learning_rate_value
        metadata["training_quantized_module"] = self.training_quantized_module

        # pylint: disable-next=protected-access
        metadata["_weights_encrypted_fit"] = self._weights_encrypted_fit
        # pylint: disable-next=protected-access
        metadata["_bias_encrypted_fit"] = self._bias_encrypted_fit

        # Concrete ML
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["loss"] = self.loss
        metadata["penalty"] = self.penalty
        metadata["alpha"] = self.alpha
        metadata["l1_ratio"] = self.l1_ratio
        metadata["fit_intercept"] = self.fit_intercept
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["shuffle"] = self.shuffle
        metadata["verbose"] = self.verbose
        metadata["epsilon"] = self.epsilon
        metadata["n_jobs"] = self.n_jobs
        metadata["random_state"] = self.random_state
        metadata["learning_rate"] = self.learning_rate
        metadata["eta0"] = self.eta0
        metadata["power_t"] = self.power_t
        metadata["early_stopping"] = self.early_stopping
        metadata["validation_fraction"] = self.validation_fraction
        metadata["n_iter_no_change"] = self.n_iter_no_change
        metadata["class_weight"] = self.class_weight
        metadata["warm_start"] = self.warm_start
        metadata["average"] = self.average

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata.pop("n_bits"))

        # Concrete ML for training in FHE
        obj.n_bits_training = metadata.pop("n_bits_training")
        obj.rounding_training = metadata.pop("rounding_training")
        obj.fit_encrypted = metadata.pop("fit_encrypted")
        obj.parameters_range = metadata.pop("parameters_range")
        obj.batch_size = metadata.pop("batch_size")
        obj.learning_rate_value = metadata.pop("learning_rate_value")
        obj.training_quantized_module = metadata.pop("training_quantized_module")

        # pylint: disable-next=protected-access
        obj._weights_encrypted_fit = metadata.pop("_weights_encrypted_fit")
        # pylint: disable-next=protected-access
        obj._bias_encrypted_fit = metadata.pop("_bias_encrypted_fit")

        obj.random_number_generator = numpy.random.default_rng(metadata["random_state"])

        # Concrete ML
        obj.sklearn_model = metadata.pop("sklearn_model")
        obj._is_fitted = metadata.pop("_is_fitted")
        obj._is_compiled = metadata.pop("_is_compiled")
        obj.input_quantizers = metadata.pop("input_quantizers")
        obj.output_quantizers = metadata.pop("output_quantizers")
        obj._weight_quantizer = metadata.pop("_weight_quantizer")
        obj.onnx_model_ = metadata.pop("onnx_model_")
        obj._q_weights = metadata.pop("_q_weights")
        obj._q_bias = metadata.pop("_q_bias")
        obj.post_processing_params = metadata.pop("post_processing_params")

        # scikit-learn
        obj.loss = metadata.pop("loss")
        obj.penalty = metadata.pop("penalty")
        obj.alpha = metadata.pop("alpha")
        obj.l1_ratio = metadata.pop("l1_ratio")
        obj.fit_intercept = metadata.pop("fit_intercept")
        obj.max_iter = metadata.pop("max_iter")
        obj.tol = metadata.pop("tol")
        obj.shuffle = metadata.pop("shuffle")
        obj.verbose = metadata.pop("verbose")
        obj.epsilon = metadata.pop("epsilon")
        obj.n_jobs = metadata.pop("n_jobs")
        obj.random_state = metadata.pop("random_state")
        obj.learning_rate = metadata.pop("learning_rate")
        obj.eta0 = metadata.pop("eta0")
        obj.power_t = metadata.pop("power_t")
        obj.early_stopping = metadata.pop("early_stopping")
        obj.validation_fraction = metadata.pop("validation_fraction")
        obj.n_iter_no_change = metadata.pop("n_iter_no_change")
        obj.class_weight = metadata.pop("class_weight")
        obj.warm_start = metadata.pop("warm_start")
        obj.average = metadata.pop("average")

        # Check that all attributes found in metadata were loaded
        assert len(metadata) == 0

        return obj


class SGDRegressor(SklearnSGDRegressorMixin):
    """An FHE linear regression model fitted with stochastic gradient descent.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on SGDRegressor please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
    """

    sklearn_model_class = sklearn.linear_model.SGDRegressor

    _is_a_public_cml_model = True

    def __init__(  # pylint: disable=too-many-arguments
        self,
        n_bits=8,
        loss="squared_error",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):

        super().__init__(n_bits=n_bits)

        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["loss"] = self.loss
        metadata["penalty"] = self.penalty
        metadata["alpha"] = self.alpha
        metadata["l1_ratio"] = self.l1_ratio
        metadata["fit_intercept"] = self.fit_intercept
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["shuffle"] = self.shuffle
        metadata["verbose"] = self.verbose
        metadata["epsilon"] = self.epsilon
        metadata["random_state"] = self.random_state
        metadata["learning_rate"] = self.learning_rate
        metadata["eta0"] = self.eta0
        metadata["power_t"] = self.power_t
        metadata["early_stopping"] = self.early_stopping
        metadata["validation_fraction"] = self.validation_fraction
        metadata["n_iter_no_change"] = self.n_iter_no_change
        metadata["warm_start"] = self.warm_start
        metadata["average"] = self.average

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        obj.loss = metadata["loss"]
        obj.penalty = metadata["penalty"]
        obj.alpha = metadata["alpha"]
        obj.l1_ratio = metadata["l1_ratio"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.shuffle = metadata["shuffle"]
        obj.verbose = metadata["verbose"]
        obj.epsilon = metadata["epsilon"]
        obj.random_state = metadata["random_state"]
        obj.learning_rate = metadata["learning_rate"]
        obj.eta0 = metadata["eta0"]
        obj.power_t = metadata["power_t"]
        obj.early_stopping = metadata["early_stopping"]
        obj.validation_fraction = metadata["validation_fraction"]
        obj.n_iter_no_change = metadata["n_iter_no_change"]
        obj.warm_start = metadata["warm_start"]
        obj.average = metadata["average"]

        return obj


class ElasticNet(SklearnLinearRegressorMixin):
    """An ElasticNet regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on ElasticNet please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    """

    sklearn_model_class = sklearn.linear_model.ElasticNet
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["alpha"] = self.alpha
        metadata["l1_ratio"] = self.l1_ratio
        metadata["fit_intercept"] = self.fit_intercept
        metadata["copy_X"] = self.copy_X
        metadata["positive"] = self.positive
        metadata["precompute"] = self.precompute
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["warm_start"] = self.warm_start
        metadata["random_state"] = self.random_state
        metadata["selection"] = self.selection

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # scikit-learn
        obj.alpha = metadata["alpha"]
        obj.l1_ratio = metadata["l1_ratio"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.copy_X = metadata["copy_X"]
        obj.positive = metadata["positive"]
        obj.precompute = metadata["precompute"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.warm_start = metadata["warm_start"]
        obj.random_state = metadata["random_state"]
        obj.selection = metadata["selection"]

        return obj


class Lasso(SklearnLinearRegressorMixin):
    """A Lasso regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on Lasso please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    sklearn_model_class = sklearn.linear_model.Lasso
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        alpha: float = 1.0,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.selection = selection
        self.tol = tol
        self.precompute = precompute
        self.random_state = random_state

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["copy_X"] = self.copy_X
        metadata["positive"] = self.positive
        metadata["max_iter"] = self.max_iter
        metadata["warm_start"] = self.warm_start
        metadata["selection"] = self.selection
        metadata["tol"] = self.tol
        metadata["precompute"] = self.precompute
        metadata["random_state"] = self.random_state

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # scikit-learn
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.copy_X = metadata["copy_X"]
        obj.positive = metadata["positive"]
        obj.max_iter = metadata["max_iter"]
        obj.warm_start = metadata["warm_start"]
        obj.selection = metadata["selection"]
        obj.tol = metadata["tol"]
        obj.precompute = metadata["precompute"]
        obj.random_state = metadata["random_state"]

        return obj


class Ridge(SklearnLinearRegressorMixin):
    """A Ridge regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on Ridge please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """

    sklearn_model_class = sklearn.linear_model.Ridge
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        n_bits=8,
        alpha: float = 1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=0.0001,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        # Call SklearnLinearModelMixin's __init__ method
        super().__init__(n_bits=n_bits)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["alpha"] = self.alpha
        metadata["fit_intercept"] = self.fit_intercept
        metadata["copy_X"] = self.copy_X
        metadata["positive"] = self.positive
        metadata["max_iter"] = self.max_iter
        metadata["tol"] = self.tol
        metadata["solver"] = self.solver
        metadata["random_state"] = self.random_state

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # scikit-learn
        obj.alpha = metadata["alpha"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.copy_X = metadata["copy_X"]
        obj.positive = metadata["positive"]
        obj.max_iter = metadata["max_iter"]
        obj.tol = metadata["tol"]
        obj.solver = metadata["solver"]
        obj.random_state = metadata["random_state"]
        return obj


class LogisticRegression(SklearnLinearClassifierMixin):
    """A logistic regression model with FHE.

    Parameters:
        n_bits (int, Dict[str, int]): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
            Default to 8.

    For more details on LogisticRegression please refer to the scikit-learn documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    sklearn_model_class = sklearn.linear_model.LogisticRegression
    _is_a_public_cml_model = True

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        # Concrete ML specific arguments
        n_bits=8,
        # scikit-learn arguments
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        # Call BaseClassifier's __init__ method
        super().__init__(n_bits=n_bits)

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def dump_dict(self) -> Dict[str, Any]:
        assert self._weight_quantizer is not None, self._is_not_fitted_error_message()

        metadata: Dict[str, Any] = {}

        # Concrete ML
        metadata["n_bits"] = self.n_bits
        metadata["sklearn_model"] = self.sklearn_model
        metadata["_is_fitted"] = self._is_fitted
        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["_weight_quantizer"] = self._weight_quantizer
        metadata["output_quantizers"] = self.output_quantizers
        metadata["onnx_model_"] = self.onnx_model_
        metadata["_q_weights"] = self._q_weights
        metadata["_q_bias"] = self._q_bias
        metadata["post_processing_params"] = self.post_processing_params

        # scikit-learn
        metadata["penalty"] = self.penalty
        metadata["dual"] = self.dual
        metadata["tol"] = self.tol
        metadata["C"] = self.C
        metadata["fit_intercept"] = self.fit_intercept
        metadata["intercept_scaling"] = self.intercept_scaling
        metadata["class_weight"] = self.class_weight
        metadata["random_state"] = self.random_state
        metadata["solver"] = self.solver
        metadata["max_iter"] = self.max_iter
        metadata["multi_class"] = self.multi_class
        metadata["verbose"] = self.verbose
        metadata["warm_start"] = self.warm_start
        metadata["n_jobs"] = self.n_jobs
        metadata["l1_ratio"] = self.l1_ratio

        return metadata

    @classmethod
    def load_dict(cls, metadata: Dict):

        # Instantiate the model
        obj = cls(n_bits=metadata["n_bits"])

        # Concrete ML
        obj.sklearn_model = metadata["sklearn_model"]
        obj._is_fitted = metadata["_is_fitted"]
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._weight_quantizer = metadata["_weight_quantizer"]
        obj.onnx_model_ = metadata["onnx_model_"]
        obj._q_weights = metadata["_q_weights"]
        obj._q_bias = metadata["_q_bias"]
        obj.post_processing_params = metadata["post_processing_params"]

        # scikit-learn
        obj.penalty = metadata["penalty"]
        obj.dual = metadata["dual"]
        obj.tol = metadata["tol"]
        obj.C = metadata["C"]
        obj.fit_intercept = metadata["fit_intercept"]
        obj.intercept_scaling = metadata["intercept_scaling"]
        obj.class_weight = metadata["class_weight"]
        obj.random_state = metadata["random_state"]
        obj.solver = metadata["solver"]
        obj.max_iter = metadata["max_iter"]
        obj.multi_class = metadata["multi_class"]
        obj.verbose = metadata["verbose"]
        obj.warm_start = metadata["warm_start"]
        obj.n_jobs = metadata["n_jobs"]
        obj.l1_ratio = metadata["l1_ratio"]

        return obj


# pylint: enable=too-many-instance-attributes,invalid-name
