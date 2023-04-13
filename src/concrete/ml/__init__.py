"""ML module."""
import os

from .version import __version__

# If the use of skops needs to be disabled.
# This could be useful when loading a model with Python 3.7 with a higher version
USE_SKOPS = int(os.environ.get("USE_SKOPS", 1))

# These are all the trusted types that are considered by skops
TRUSTED_SKOPS = [
    "numpy.int64",
    "numpy.float64",
    "numpy.int32",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRegressor",
    "sklearn._loss.glm_distribution.DistributionBoundary",
    "sklearn._loss.glm_distribution.TweedieDistribution",
    "sklearn._loss.glm_distribution.GammaDistribution",
    "sklearn._loss.glm_distribution.PoissonDistribution",
    "sklearn.linear_model._glm.link.LogLink",
    "sklearn.linear_model._glm.link.IdentityLink",
    "sklearn._loss.link.IdentityLink",
    "sklearn._loss.link.Interval",
    "sklearn._loss.link.LogLink",
    "sklearn._loss.link.LogLink",
    "sklearn._loss._loss.CyHalfTweedieLossIdentity",
    "sklearn._loss.loss.HalfTweedieLossIdentity",
    "sklearn._loss._loss.CyHalfPoissonLoss",
    "sklearn._loss.loss.HalfPoissonLoss",
    "sklearn._loss._loss.CyHalfGammaLoss",
    "sklearn._loss.loss.HalfGammaLoss",
    "sklearn._loss._loss.CyHalfTweedieLoss",
    "sklearn._loss.loss.HalfTweedieLoss",
]

# If USE_SKOPS is False or skops can't be imported we default to pickle
try:
    if USE_SKOPS:
        from skops.io import dumps as dumps_sklearn
        from skops.io import loads as loads_sklearn
    else:  # pragma: no cover
        raise ImportError()
except ImportError:  # pragma: no cover
    USE_SKOPS = False
    from pickle import dumps as dumps_sklearn
    from pickle import loads as loads_sklearn
