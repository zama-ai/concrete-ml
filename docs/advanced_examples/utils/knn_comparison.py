"""This file runs `classifier_comparison_utils.py` script for the KNN classifier, because it doesn't
work on Jupyter for now. This file must be deleted once:
https://github.com/zama-ai/concrete-ml-internal/issues/4018 is solved."""

from functools import partial

from classifier_comparison_utils import make_classifier_comparison

from concrete.ml.sklearn import KNeighborsClassifier

if __name__ == "__main__":
    knn_classifiers = [
        (partial(KNeighborsClassifier, n_bits=2, n_neighbors=3), "3nn"),
        #(partial(KNeighborsClassifier, n_bits=4, n_neighbors=5), "5nn"),
    ]

    # pylint: disable-next=unexpected-keyword-arg
    make_classifier_comparison(
        "KNN_Classifier", knn_classifiers, 0, verbose=True, show_score=False, save_plot=True
    )
