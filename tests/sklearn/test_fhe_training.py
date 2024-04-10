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
def test_init_warning_error_raises(n_bits, parameter_min_max):
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


def test_clear_fit_warning_error_raises():
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

    # Initialize the model
    with warnings.catch_warnings():
        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model_disable = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            early_stopping=False,
            fit_intercept=fit_intercept,
            verbose=True,
        )
        model_disable.training_p_error = 1e-15

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )

        # Fit the model in the clear
        model_disable.fit(x, y, fhe="disable")

    y_pred_class_disable = model_disable.predict(x)
    y_pred_proba_disable = model_disable.predict_proba(x)
    weights_disable = (
        model_disable._weights_encrypted_fit.copy()
    )  # pylint: disable=protected-access
    bias_disable = model_disable._bias_encrypted_fit.copy()  # pylint: disable=protected-access

    # Check that we overfit properly a linearly separable dataset
    check_accuracy(y, y_pred_class_disable, threshold=0.95)

    # We need to re-create an object to avoid any issue with random state
    # Initialize the model
    with warnings.catch_warnings():
        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")
        model_simulate = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            early_stopping=False,
            fit_intercept=fit_intercept,
            verbose=True,
        )
        # We need to lower the p-error to make sure that the test passes
        model_simulate.training_p_error = 1e-15

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )

        # Fit the model with simulation
        model_simulate.fit(x, y, fhe="simulate")

    # We need to re-create an object to avoid any issue with random state
    # Initialize the model
    with warnings.catch_warnings():
        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")
        model_early_break = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            early_stopping=True,
            tol=1e100,  # Crazy high tolerance
            fit_intercept=fit_intercept,
            verbose=True,
        )
        model_early_break.training_p_error = 1e-15

    # We don't have any way to detect early break
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )

        # Fit the model with simulation
        model_early_break.fit(x, y, fhe="simulate")

    y_pred_class_simulate = model_simulate.predict(x)
    y_pred_proba_simulate = model_simulate.predict_proba(x)

    weights_simulate = (
        model_simulate._weights_encrypted_fit.copy()
    )  # pylint: disable=protected-access
    bias_simulate = model_simulate._bias_encrypted_fit.copy()  # pylint: disable=protected-access

    # Check that we overfit properly a linearly separable dataset
    check_accuracy(y, y_pred_class_simulate, threshold=0.95)

    # Make sure weight, bias and prediction values are identical between clear and
    # simulated training
    assert array_allclose_and_same_shape(weights_simulate, weights_disable)
    assert array_allclose_and_same_shape(bias_simulate, bias_disable)
    assert array_allclose_and_same_shape(y_pred_proba_simulate, y_pred_proba_disable)
    assert array_allclose_and_same_shape(y_pred_class_simulate, y_pred_class_disable)

    # Initialize a model for partial fitting
    with warnings.catch_warnings():
        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model_partial = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            warm_start=True,
            early_stopping=False,
            fit_intercept=fit_intercept,
            verbose=True,
        )
        # We need to lower the p-error to make sure that the test passes
        model_partial.training_p_error = 1e-15

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )

        for index in range(max_iter):
            # Check that we can swap between modes without any impact on
            # the final training performance
            if index % 2 == 0:
                model_partial.partial_fit(x, y, fhe="disable")
            else:
                model_partial.partial_fit(x, y, fhe="simulate")

    y_pred_class_partial = model_partial.predict(x)
    y_pred_proba_partial = model_partial.predict_proba(x)

    # pylint: disable-next=protected-access
    weights_partial = model_partial._weights_encrypted_fit.copy()
    bias_partial = model_partial._bias_encrypted_fit.copy()  # pylint: disable=protected-access

    # Check that we overfit properly a linearly separable dataset
    check_accuracy(y, y_pred_class_partial, threshold=0.95)

    # Initialize another model for partial fitting
    with warnings.catch_warnings():
        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model_partial_2 = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            warm_start=True,
            early_stopping=False,
            fit_intercept=fit_intercept,
            verbose=True,
        )
        # We need to lower the p-error to make sure that the test passes
        model_partial_2.training_p_error = 1e-15

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )
        half_iter = max_iter // 2
        other_iter = max_iter - half_iter
        model_partial_2.max_iter = half_iter
        model_partial_2.fit(x, y, fhe="disable")
        model_partial_2.max_iter = other_iter
        model_partial_2.fit(x, y, fhe="simulate")
        assert half_iter + other_iter == max_iter

    # pylint: disable-next=protected-access
    weights_partial_2 = model_partial_2._weights_encrypted_fit.copy()
    bias_partial_2 = model_partial_2._bias_encrypted_fit.copy()  # pylint: disable=protected-access

    # Make sure weight, bias and prediction values are identical between clear and
    # partial clear training
    assert array_allclose_and_same_shape(weights_disable, weights_partial)
    assert array_allclose_and_same_shape(bias_disable, bias_partial)
    assert array_allclose_and_same_shape(weights_partial_2, weights_partial)
    assert array_allclose_and_same_shape(bias_partial_2, bias_partial)
    assert array_allclose_and_same_shape(y_pred_proba_disable, y_pred_proba_partial)
    assert array_allclose_and_same_shape(y_pred_class_disable, y_pred_class_partial)

    if not fit_intercept:
        assert bias_partial == numpy.zeros((1, 1))
        assert bias_disable == numpy.zeros((1, 1))
        assert bias_simulate == numpy.zeros((1, 1))

    # Initialize a model for fitting
    with warnings.catch_warnings():
        # FHE training is an experimental feature and a warning is raised each time `fit_encrypted`
        # is set to True
        warnings.filterwarnings("ignore", message="FHE training is an experimental feature.*")

        model_from_another = SGDClassifier(
            n_bits=n_bits,
            fit_encrypted=True,
            random_state=random_state,
            parameters_range=parameters_range,
            max_iter=max_iter,
            warm_start=True,
            early_stopping=False,
            fit_intercept=fit_intercept,
            verbose=True,
        )
        model_from_another.training_p_error = 1e-15

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ONNX Preprocess - Removing mutation from node aten::sub_ on block input.*",
        )

        model_from_another.fit(
            x,
            y,
            fhe="disable",
            coef_init=model_partial.coef_,
            intercept_init=model_partial.intercept_,
        )
