"""Tests common to all sklearn models."""
import numpy
import pytest

from concrete.ml.sklearn import DecisionTreeClassifier, RandomForestClassifier, XGBClassifier


@pytest.mark.parametrize(
    "alg",
    [
        pytest.param(
            lambda random_state: RandomForestClassifier(random_state=random_state),
            id="RandomForestClassifier",
        ),
        pytest.param(
            lambda random_state: DecisionTreeClassifier(random_state=random_state),
            id="DecisionTreeClassifier",
        ),
        pytest.param(
            lambda random_state: XGBClassifier(random_state=random_state), id="XGBClassifier"
        ),
    ],
)
@pytest.mark.parametrize(
    "make_data",
    [
        pytest.param(
            lambda n_examples, n_features: (
                numpy.random.rand(n_examples, n_features),
                numpy.random.randint(0, 2, size=(n_examples,)),
            )
        ),
    ],
)
def test_random_state_fit_benchmark(make_data, alg):
    """Tests the random_state parameter."""
    random_state_constructor = numpy.random.randint(0, 2**15)
    random_state_user = numpy.random.randint(0, 2**15)

    x, y = make_data(100, 10)

    # First case: user gives his own random_state
    model = alg(random_state=random_state_constructor)
    model, sklearn_model = model.fit_benchmark(x, y, random_state=random_state_user)
    assert (
        model.random_state == random_state_user and sklearn_model.random_state == random_state_user
    )

    # Second case: user does not give random_state but seeds the constructor
    model = alg(random_state=random_state_constructor)
    model, sklearn_model = model.fit_benchmark(x, y)
    assert (model.random_state == random_state_constructor) and (
        sklearn_model.random_state == random_state_constructor
    )

    # Third case: user does not provide any seed
    model = alg(random_state=None)
    assert model.random_state is None
    model, sklearn_model = model.fit_benchmark(x, y)
    # model.random_state and sklearn_model.random_state should now be seeded with the same value
    assert model.random_state is not None and sklearn_model.random_state is not None
    assert model.random_state == sklearn_model.random_state
