# Source:
#   https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Modified to integrate Concrete-ML functions by Zama
# License: BSD 3 clause

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import time

import matplotlib.pyplot as plt
import numpy as np
from concrete.numpy.compilation.configuration import Configuration
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from concrete.ml.sklearn import DecisionTreeClassifier, NeuralNetClassifier

# The Virtual Library allows to simulate FHE executions without paying the cost of FHE
# computations. However, data is not encrypted when using the Virtual Library, the model executes
# an integer model on data in the clear.
ALWAYS_USE_VL = False

# Initialize the configuration used for Virtual Library tests only.
COMPIL_CONFIG_VL = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,
    p_error=None,  # To avoid any confusion: we are always using kwarg p_error
    global_p_error=None,  # To avoid any confusion: we are always using kwarg global_p_error
)

from IPython.display import clear_output


# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def make_classifier_comparison(title, classifiers, decision_level):

    h = 0.04  # Step size in the mesh

    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        make_moons(n_samples=200, noise=0.3, random_state=0),
        make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]

    font_size_text = 20

    fig, axs = plt.subplots(len(datasets), 2 * len(classifiers) + 1, figsize=(32, 16))
    fig.suptitle(title, fontsize=20)
    fig.patch.set_facecolor("white")

    # Iterate over datasets
    for i, dataset in enumerate(datasets):
        # Preprocess dataset
        X, y = dataset
        X = X.astype(np.float32)
        X = StandardScaler().fit_transform(X)

        # Split the data into training and test sets
        # Use 15 percent (30 points for a dataset of 200 points) for prediction
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = axs[i, 0]
        if i == 0:
            ax.set_title("Input data", fontsize=font_size_text)

        # Plot the training points
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k",
            label="Train data",
        )

        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            marker="D",
            c=y_test,
            cmap=cm_bright,
            alpha=0.6,
            edgecolors="k",
            label="Test data",
        )
        ax.legend()

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        # Iterate over the given classifiers
        for j, (classifier, model_name) in enumerate(classifiers):
            # Instantiate the model
            model = classifier()

            # Train the model and retrieve both the Concrete-ML model and its equivalent one from
            # scikit-learn
            concrete_model, sklearn_model = model.fit_benchmark(X_train, y_train)

            # Compute the predictions in clear using the scikit-learn model
            sklearn_y_pred = sklearn_model.predict(X_test)

            # Compile the Concrete-ML model
            circuit = concrete_model.compile(
                X_train,
                use_virtual_lib=ALWAYS_USE_VL,
                configuration=COMPIL_CONFIG_VL if ALWAYS_USE_VL else None,
            )

            # If the prediction are done in FHE, generate the key
            if not ALWAYS_USE_VL:

                print(
                    "Generating a key for a "
                    f"{circuit.graph.maximum_integer_bit_width()}-bit circuit"
                )

                time_begin = time.time()
                circuit.client.keygen(force=False)
                print(f"Key generation time: {time.time() - time_begin:.4f} seconds")

            # Compute the predictions in FHE using the Concrete-ML model
            time_begin = time.time()
            concrete_y_pred = concrete_model.predict(X_test, execute_in_fhe=True)
            print(
                f"Execution time: {(time.time() - time_begin) / len(X_test):.4f} "
                "seconds per sample"
            )

            # Measure the accuracy scores
            sklearn_score = accuracy_score(sklearn_y_pred, y_test)
            concrete_score = accuracy_score(concrete_y_pred, y_test)

            is_a_tree_based_model = concrete_model.__class__ in [
                DecisionTreeClassifier,
                DecisionTreeClassifier,
            ]

            # Compile the Concrete-ML model with the Virtual Library to evaluate
            # the domain grid
            circuit = concrete_model.compile(
                X_train,
                use_virtual_lib=True,
                configuration=COMPIL_CONFIG_VL,
            )

            # If the model is not a tree-based model, retrieve the maximum integer bitwidth
            # reached within its circuit.
            bitwidth = None
            if not is_a_tree_based_model:
                bitwidth = circuit.graph.maximum_integer_bit_width()

            # Plot the decision boundaries.
            # For that, a color is assigned to each point in the mesh, which is obtained as a
            # cartesian product of [x_min, x_max] with [y_min, y_max].
            if hasattr(sklearn_model, "decision_function"):
                # Note that although we pass execute_in_fhe=True, execution is only simulated
                # since we compiled with use_virtual_lib=True
                sklearn_Z = sklearn_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
                concrete_Z = concrete_model.decision_function(
                    np.c_[xx.ravel(), yy.ravel()],
                    execute_in_fhe=True,
                )
            else:
                # Note that although we pass execute_in_fhe=True, execution is only simulated
                # since we compiled with use_virtual_lib=True
                sklearn_Z = sklearn_model.predict_proba(
                    np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
                )[:, 1]
                concrete_Z = concrete_model.predict_proba(
                    np.c_[xx.ravel(), yy.ravel()],
                    execute_in_fhe=True,
                )[:, 1]

            for k, (framework, score, Z) in enumerate(
                zip(
                    ["scikit-learn", "Concrete-ML"],
                    [sklearn_score, concrete_score],
                    [sklearn_Z, concrete_Z],
                )
            ):
                ax = axs[i, 2 * j + k + 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

                # Plot the testing points
                ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    marker="D",
                    cmap=cm_bright,
                    edgecolors="k",
                    alpha=0.6,
                )

                ax.contour(
                    xx,
                    yy,
                    Z,
                    levels=[decision_level],
                    linewidths=2,
                )

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())

                if i == 0:
                    ax.set_title(model_name + f" ({framework})", fontsize=font_size_text)

                ax.text(
                    xx.max() - 0.3,
                    yy.min() + 0.3,
                    f"{score*100:0.1f}%",
                    size=font_size_text,
                    horizontalalignment="right",
                )

                if bitwidth and framework == "Concrete-ML":
                    ax.text(
                        xx.max() - 0.3,
                        yy.min() + 1.0,
                        f"bitwidth={bitwidth}",
                        size=font_size_text,
                        horizontalalignment="right",
                    )

    clear_output(wait=True)

    plt.tight_layout()
    plt.show()
