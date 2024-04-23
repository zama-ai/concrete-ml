"""Tests training in FHE."""

import re
import warnings

import numpy
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from concrete.ml.common.utils import array_allclose_and_same_shape
from concrete.ml.sklearn import SGDClassifier


def get_blob_data(binary_targets=True, scale_input=False, parameters_range=None):
    """Get the training data."""

    n_samples = 1000
    n_features = 8

    # Determine the number of target classes to generate
    centers = 2 if binary_targets else 3

    # Generate the input and target values
    # pylint: disable-next=unbalanced-tuple-unpacking
    x, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features)

    # Scale the input values if needed
    if scale_input:
        assert parameters_range is not None

        preprocessing = MinMaxScaler(feature_range=parameters_range)
        x = preprocessing.fit_transform(x)

    return x, y


@pytest.mark.parametrize("n_bits, parameter_min_max", [pytest.param(7, 1.0)])
def test_init_error_raises(n_bits, parameter_min_max):
    """Test that initializing the model for FHE training using wrong parameters raises errors."""

    # Model parameters
    random_state = numpy.random.randint(0, 2**15)
    parameters_range = (-parameter_min_max, parameter_min_max)

    with pytest.warns(
        UserWarning,
        match=(
            "FHE training is an experimental feature. Please be aware that the API might change "
            "in future versions."
        ),
    ):
        SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
        )

    with warnings.catch_warnings():

        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Only 'log_loss' is currently supported if FHE training is enabled"
                " (fit_encrypted=True). Got loss='perceptron'"
            ),
        ):
            SGDClassifier(
                n_bits=n_bits,
                fit_encrypted=True,
                loss="perceptron",
                random_state=random_state,
                parameters_range=parameters_range,
            )

        with pytest.raises(
            ValueError, match="Setting 'parameter_range' is mandatory if FHE training is enabled."
        ):
            SGDClassifier(
                n_bits=n_bits,
                fit_encrypted=True,
                random_state=random_state,
                parameters_range=None,
                fit_intercept=True,
            )

            SGDClassifier(
                n_bits=n_bits,
                fit_encrypted=True,
                random_state=random_state,
                parameters_range=parameters_range,
                fit_intercept=False,
            )


@pytest.mark.parametrize("n_bits, max_iter, parameter_min_max", [pytest.param(7, 30, 1.0)])
def test_fit_error_if_non_binary_targets(n_bits, max_iter, parameter_min_max):
    """Test that training in FHE on a data-set with more than 2 target classes raises an error."""

    # Model parameters
    random_state = numpy.random.randint(0, 2**15)
    parameters_range = (-parameter_min_max, parameter_min_max)

    # Generate a data-set with three target classes
    x, y = get_blob_data(binary_targets=False)

    with warnings.catch_warnings():

        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
        )

    with pytest.raises(
        NotImplementedError,
        match="Only binary classification is currently supported when FHE training is enabled.*",
    ):
        model.fit(x, y, fhe="disable")

    with pytest.raises(
        NotImplementedError,
        match="Only binary classification is currently supported when FHE training is enabled.*",
    ):
        model.partial_fit(x, y, fhe="disable")


def test_clear_fit_error_raises():
    """Test that training in clear using wrong parameters raises proper errors."""

    # Model parameters
    parameters_range = (-1.0, 1.0)

    # Generate a data-set
    x, y = get_blob_data()

    with pytest.raises(NotImplementedError, match="Only one of .* loss is supported.*"):
        model = SGDClassifier(
            fit_encrypted=False,
            parameters_range=parameters_range,
            loss="perceptron",
        )

    model = SGDClassifier(fit_encrypted=False, parameters_range=parameters_range)

    with pytest.raises(
        ValueError, match="Parameter 'fhe' should not be set when FHE training is disabled.*"
    ):
        model.fit(x, y, fhe="disable")

    with pytest.raises(
        NotImplementedError, match="Partial fit is not currently supported for clear training."
    ):
        model.partial_fit(x, y, fhe=None)


@pytest.mark.parametrize("n_bits, max_iter, parameter_min_max", [pytest.param(7, 30, 1.0)])
def test_encrypted_fit_warning_error_raises(n_bits, max_iter, parameter_min_max):
    """Test that training in FHE using wrong parameters properly raises some errors."""

    # Model parameters
    random_state = numpy.random.randint(0, 2**15)
    parameters_range = (-parameter_min_max, parameter_min_max)

    # Generate a data-set with binary target classes
    x, y = get_blob_data(scale_input=True, parameters_range=parameters_range)

    with warnings.catch_warnings():

        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
        )

    with pytest.warns(
        UserWarning,
        match="Parameter 'fhe' isn't set while FHE training is enabled.\n"
        "Defaulting to 'fhe='disable''",
    ):
        model.fit(x, y, fhe=None)

    with pytest.warns(
        UserWarning,
        match="Parameter 'fhe' isn't set while FHE training is enabled.\n"
        "Defaulting to 'fhe='disable''",
    ):
        model.partial_fit(x, y, fhe=None)

    with pytest.raises(
        NotImplementedError,
        match="Parameter 'sample_weight' is currently not supported for FHE training.",
    ):
        model.fit(x, y, fhe="disable", sample_weight=numpy.ones((1,)))

    x_3d = numpy.expand_dims(x, axis=-1)

    with pytest.raises(NotImplementedError, match="Input values must be 2D.*"):
        model.fit(x_3d, y, fhe="disable")

    with pytest.raises(NotImplementedError, match="Input values must be 2D.*"):
        model.partial_fit(x_3d, y, fhe="disable")

    y_2d = numpy.expand_dims(y, axis=-1)

    with pytest.raises(NotImplementedError, match="Target values must be 1D.*"):
        model.fit(x, y_2d, fhe="disable")

    with pytest.raises(NotImplementedError, match="Target values must be 1D.*"):
        model.partial_fit(x, y_2d, fhe="disable")

    with pytest.warns(
        UserWarning,
        match="FHE training is an experimental feature. "
        "Please be aware that the API might change in future versions.",
    ):
        model = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            loss="log_loss",
        )

    with pytest.warns(UserWarning, match="ONNX Preprocess - Removing mutation from node .*"):
        model.fit(x, y, fhe="disable")

    with pytest.raises(NotImplementedError, match=""):
        model.loss = "random"
        model.predict_proba(x)

    with pytest.warns(
        UserWarning,
        match="FHE training is an experimental feature. "
        "Please be aware that the API might change in future versions.",
    ):
        model = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            loss="log_loss",
        )

    assert isinstance(y, numpy.ndarray)

    with pytest.raises(
        NotImplementedError,
        match="Parameter 'sample_weight' is currently not supported for FHE training.",
    ):
        model.fit(x, y, sample_weight=y + 1, fhe="disable")


@pytest.mark.parametrize("loss", ["log_loss", "modified_huber"])
@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("n_bits, max_iter, parameter_min_max", [pytest.param(8, 100, 1.0)])
def test_clear_fit(
    loss,
    binary,
    n_bits,
    max_iter,
    parameter_min_max,
):
    """Test that encrypted fitting works properly."""

    # Model parameters
    random_state = numpy.random.randint(0, 2**15)
    parameters_range = (-parameter_min_max, parameter_min_max)

    # Generate a data-set
    x, y = get_blob_data(binary_targets=binary, scale_input=True, parameters_range=parameters_range)

    random_state = numpy.random.randint(0, 2**15)

    model = SGDClassifier(
        n_bits=n_bits,
        fit_encrypted=False,
        random_state=random_state,
        max_iter=max_iter,
        early_stopping=False,
        fit_intercept=True,
        loss=loss,
    )

    model.fit(x, y)
    model.predict(x)
    model.predict_proba(x)
    model.compile(x)
    model.predict(x, fhe="simulate")
    model.predict_proba(x, fhe="simulate")


# pylint: disable=too-many-arguments, too-many-branches
def check_encrypted_fit(
    x,
    y,
    n_bits,
    random_state,
    parameters_range,
    max_iter,
    fit_intercept,
    check_accuracy=None,
    fhe=None,
    partial_fit=False,
    warm_fit=False,
    random_number_generator=None,
    init_kwargs=None,
    fit_kwargs=None,
):
    """Check that encrypted fitting works properly."""

    # Fit method cannot be partial or warm type as the same time
    assert not (partial_fit and warm_fit)

    # fhe mode cannot be set when executing partial or warm fitting, but must be set otherwise
    if partial_fit or warm_fit:
        assert fhe is None
    else:
        assert fhe is not None

    if init_kwargs is None:
        init_kwargs = {}

    if fit_kwargs is None:
        fit_kwargs = {}

    # Initialize the model
    with warnings.catch_warnings():

        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            verbose=True,
            **init_kwargs,
        )

        # If a RNG instance if provided, use it to set the new model's one
        if random_number_generator is not None:
            model.random_number_generator = random_number_generator

        # We need to lower the p-error to make sure that the test passes
        model.training_p_error = 1e-15

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )

        if partial_fit:

            # Check that we can swap between disable and simulation modes without any impact on the
            # final training performance
            for index in range(max_iter):
                if index % 2 == 0:
                    model.partial_fit(x, y, fhe="disable")
                else:
                    model.partial_fit(x, y, fhe="simulate")

        elif warm_fit:

            # Check that we can swap between disable and simulation modes without any impact on the
            # final training performance
            half_iter = max_iter // 2
            model.max_iter = half_iter
            model.fit(x, y, fhe="disable")

            other_iter = max_iter - half_iter
            model.max_iter = other_iter
            model.fit(x, y, fhe="simulate")

            assert half_iter + other_iter == max_iter

        else:
            # Fit the model
            model.fit(x, y, fhe=fhe, **fit_kwargs)

    y_pred_class = model.predict(x)
    y_pred_proba = model.predict_proba(x)

    weights = model._weights_encrypted_fit.copy()  # pylint: disable=protected-access

    bias = model._bias_encrypted_fit.copy()  # pylint: disable=protected-access

    if not fit_intercept:
        assert bias == numpy.zeros(
            (1, 1)
        ), "When the model is fitted without bias, the bias term should only be made of zeros."

    # If relevant, check that we overfit properly a linearly separable dataset
    if check_accuracy is not None:
        check_accuracy(y, y_pred_class)

    return weights, bias, y_pred_proba, y_pred_class, model.random_number_generator


# pylint: disable=too-many-statements,protected-access,too-many-locals
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("label_offset", [0, 1])
@pytest.mark.parametrize("n_bits, max_iter, parameter_min_max", [pytest.param(7, 30, 1.0)])
def test_encrypted_fit_coherence(
    fit_intercept, label_offset, n_bits, max_iter, parameter_min_max, check_accuracy
):
    """Test that encrypted fitting works properly."""

    # Model parameters
    random_state = numpy.random.randint(0, 2**15)
    parameters_range = (-parameter_min_max, parameter_min_max)

    # Generate a data-set with binary target classes
    x, y = get_blob_data(scale_input=True, parameters_range=parameters_range)
    y = y + label_offset

    weights_disable, bias_disable, y_pred_proba_disable, y_pred_class_disable, _ = (
        check_encrypted_fit(
            x,
            y,
            n_bits,
            random_state,
            parameters_range,
            max_iter,
            fit_intercept,
            check_accuracy=check_accuracy,
            fhe="disable",
        )
    )

    weights_simulated, bias_simulated, y_pred_proba_simulated, y_pred_class_simulated, _ = (
        check_encrypted_fit(
            x,
            y,
            n_bits,
            random_state,
            parameters_range,
            max_iter,
            fit_intercept,
            check_accuracy=check_accuracy,
            fhe="simulate",
        )
    )

    # Make sure weight, bias and prediction values are identical between clear and
    # simulated training
    assert array_allclose_and_same_shape(weights_simulated, weights_disable)
    assert array_allclose_and_same_shape(bias_simulated, bias_disable)
    assert array_allclose_and_same_shape(y_pred_proba_simulated, y_pred_proba_disable)
    assert array_allclose_and_same_shape(y_pred_class_simulated, y_pred_class_disable)

    # Define early break parameters, with a very high tolerance
    early_break_kwargs = {"early_stopping": True, "tol": 1e100}

    # We don't have any way to properly test early break, we therefore disable the accuracy check
    # in order to avoid flaky issues
    check_encrypted_fit(
        x,
        y,
        n_bits,
        random_state,
        parameters_range,
        max_iter,
        fit_intercept,
        check_accuracy=None,
        fhe="simulate",
        init_kwargs=early_break_kwargs,
    )

    weights_partial, bias_partial, y_pred_proba_partial, y_pred_class_partial, _ = (
        check_encrypted_fit(
            x,
            y,
            n_bits,
            random_state,
            parameters_range,
            max_iter,
            fit_intercept,
            check_accuracy=check_accuracy,
            partial_fit=True,
        )
    )

    # Make sure weight, bias and prediction values are identical between clear and partial fitting
    assert array_allclose_and_same_shape(weights_disable, weights_partial)
    assert array_allclose_and_same_shape(bias_disable, bias_partial)
    assert array_allclose_and_same_shape(y_pred_proba_disable, y_pred_proba_partial)
    assert array_allclose_and_same_shape(y_pred_class_disable, y_pred_class_partial)

    # Define warm fitting parameters
    warm_fit_init_kwargs = {"warm_start": True}

    weights_warm, bias_warm, y_pred_proba_warm, y_pred_class_warm, _ = check_encrypted_fit(
        x,
        y,
        n_bits,
        random_state,
        parameters_range,
        max_iter,
        fit_intercept,
        check_accuracy=check_accuracy,
        warm_fit=True,
        init_kwargs=warm_fit_init_kwargs,
    )

    # Make sure weight, bias and prediction values are identical between clear and warm fitting
    assert array_allclose_and_same_shape(weights_disable, weights_warm)
    assert array_allclose_and_same_shape(bias_disable, bias_warm)
    assert array_allclose_and_same_shape(y_pred_proba_disable, y_pred_proba_warm)
    assert array_allclose_and_same_shape(y_pred_class_disable, y_pred_class_warm)

    first_iterations = max_iter // 2

    # Fit the model for max_iter // 2 iterations and retrieved the weight/bias values, as well as
    # the RNG object
    weights_coef_init, bias_coef_init, _, _, rng_coef_init = check_encrypted_fit(
        x,
        y,
        n_bits,
        random_state,
        parameters_range,
        first_iterations,
        fit_intercept,
        check_accuracy=check_accuracy,
        fhe="simulate",
    )

    last_iterations = max_iter - first_iterations

    # Define coef parameters
    coef_init_fit_kwargs = {
        "coef_init": weights_coef_init,
        "intercept_init": bias_coef_init,
    }

    # Fit the model for the remaining iterations starting at the previous weight/bias values. It is
    # necessary to provide the RNG object as well in order to keep data shuffle consistent
    weights_coef_init, bias_coef_init, y_pred_proba_coef_init, y_pred_class_coef_init, _ = (
        check_encrypted_fit(
            x,
            y,
            n_bits,
            random_state,
            parameters_range,
            last_iterations,
            fit_intercept,
            check_accuracy=check_accuracy,
            fhe="simulate",
            random_number_generator=rng_coef_init,
            fit_kwargs=coef_init_fit_kwargs,
        )
    )

    # Make sure weight, bias and prediction values are identical between clear fitting with and
    # without using initial weight/bias values
    assert array_allclose_and_same_shape(weights_disable, weights_coef_init)
    assert array_allclose_and_same_shape(bias_disable, bias_coef_init)
    assert array_allclose_and_same_shape(y_pred_proba_disable, y_pred_proba_coef_init)
    assert array_allclose_and_same_shape(y_pred_class_disable, y_pred_class_coef_init)
