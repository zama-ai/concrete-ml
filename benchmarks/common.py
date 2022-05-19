import random
from functools import partial

import concrete.numpy as cnp
import numpy as np
import py_progress_tracker as progress
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
)

# This is only for benchmarks to speed up compilation times
BENCHMARK_CONFIGURATION = cnp.Configuration(
    dump_artifacts_on_unexpected_failures=True,
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_keycache_location="ConcreteNumpyKeyCache",
    show_mlir=False,
    show_graph=False,
    jit=True,
)


def run_and_report_metric(y_gt, y_pred, metric, metric_id, metric_label):
    """Run a single metric and report results to progress tracker"""
    value = metric(y_gt, y_pred) if y_gt.size > 0 else 0
    progress.measure(
        id=metric_id,
        label=metric_label,
        value=value,
    )


def run_and_report_classification_metrics(y_gt, y_pred, metric_id_prefix, metric_label_prefix):
    """Run several metrics and report results to progress tracker with computed name and id"""

    metric_info = [
        (accuracy_score, "acc", "Accuracy"),
        (partial(f1_score, average="weighted"), "f1", "F1Score"),
        (matthews_corrcoef, "mcc", "MCC"),
    ]
    for (metric, metric_id, metric_label) in metric_info:
        run_and_report_metric(
            y_gt,
            y_pred,
            metric,
            "_".join((metric_id_prefix, metric_id)),
            " ".join((metric_label_prefix, metric_label)),
        )


def run_and_report_regression_metrics(y_gt, y_pred, metric_id_prefix, metric_label_prefix):
    """Run several metrics and report results to progress tracker with computed name and id"""

    metric_info = [(r2_score, "r2_score", "R2Score"), (mean_squared_error, "MSE", "MSE")]
    for (metric, metric_id, metric_label) in metric_info:
        run_and_report_metric(
            y_gt,
            y_pred,
            metric,
            "_".join((metric_id_prefix, metric_id)),
            " ".join((metric_label_prefix, metric_label)),
        )


def seed_everything(seed):
    random.seed(seed)
    seed += 1
    np.random.seed(seed % 2**32)
    seed += 1
    torch.manual_seed(seed)
    seed += 1
    torch.use_deterministic_algorithms(True)
    return seed
