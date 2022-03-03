"""Implement Support Vector Machine."""
from __future__ import annotations

import numpy
import onnx
import sklearn.linear_model

from ..onnx.onnx_model_manipulations import keep_following_outputs_discard_others
from .linear_model import SklearnLinearModelMixin


class LinearSVR(SklearnLinearModelMixin, sklearn.svm.LinearSVR):
    """A Regression Support Vector Machine (SVM)."""

    sklearn_alg = sklearn.svm.LinearSVR

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        n_bits=2,
        epsilon=0.0,
        tol=0.0001,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        super().__init__(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.n_bits = n_bits

    # pylint: enable=too-many-arguments


class LinearSVC(SklearnLinearModelMixin, sklearn.svm.LinearSVC):
    """A Classification Support Vector Machine (SVM)."""

    sklearn_alg = sklearn.svm.LinearSVC

    # pylint: disable=too-many-arguments
    # pylint: disable=invalid-name
    def __init__(
        self,
        n_bits=2,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual=True,
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        super().__init__(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            C=C,
            multi_class=multi_class,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.n_bits = n_bits

    # pylint: enable=too-many-arguments
    # pylint: enable=invalid-name

    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/425:
    # use clean_graph and predict from BaseLinearClassifierMixin
    # but difficulties because we need to make python understand that
    # LinearSVC.clean_graph must be BaseLinearClassifierMixin.clean_graph
    # and not SklearnLinearModelMixin.clean_graph
    # pylint: disable=duplicate-code
    # pylint: disable=R0801
    def clean_graph(self, onnx_model: onnx.ModelProto):
        nodes_to_remove = []
        output_to_follow = "variable"
        # Find nodes to remove (after the sigmoid)
        sigmoid_reached = False
        for node in onnx_model.graph.node:
            if sigmoid_reached:
                nodes_to_remove.append(node)
            if node.op_type == "Sigmoid":
                sigmoid_reached = True
                # Create output node

                onnx_model.graph.output[0].CopyFrom(
                    onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, [2])
                )
                output_to_follow = node.output[0]

        if sigmoid_reached:
            # Remove nodes
            for node in nodes_to_remove:
                onnx_model.graph.node.remove(node)

        keep_following_outputs_discard_others(onnx_model, [output_to_follow])
        return super().clean_graph(onnx_model)

    # pylint: disable=arguments-differ
    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/375: we need to refacto
    def decision_function(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        y_preds = super().predict(X, execute_in_fhe)
        return y_preds

    # pylint: enable=arguments-differ

    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/425:
    # use clean_graph and predict from BaseLinearClassifierMixin
    # but difficulties because we need to make python understand that
    # LinearSVC.clean_graph must be BaseLinearClassifierMixin.clean_graph
    # and not SklearnLinearModelMixin.clean_graph
    # FIXME, https://github.com/zama-ai/concrete-ml-internal/issues/375: we need to refacto
    def predict(self, X: numpy.ndarray, execute_in_fhe: bool = False) -> numpy.ndarray:
        y_preds = self.decision_function(X, execute_in_fhe)
        if y_preds.shape[1] == 1:
            # Sigmoid already applied in the graph
            y_preds = numpy.concatenate((1 - y_preds, y_preds), axis=1)
        y_preds = numpy.argmax(y_preds, axis=1)
        return y_preds

    # pylint: enable=duplicate-code
    # pylint: enable=R0801
