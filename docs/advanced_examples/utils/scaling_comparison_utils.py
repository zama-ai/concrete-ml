import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

# pylint: disable=too-many-locals
def plot_data(axs, X_train, y_train, X_test, y_test, model, name, h = 0.04, font_size_text = 20):
    # Train the model and retrieve both the Concrete ML model and its equivalent one from
    # scikit-learn
    concrete_model, sklearn_model = model.fit_benchmark(X_train, y_train)

    # Compute the predictions in clear using the scikit-learn model
    sklearn_y_pred = sklearn_model.predict(X_test)

    # Compile the Concrete ML model
    circuit = concrete_model.compile(X_test)
    concrete_y_pred = concrete_model.predict(X_test, fhe="simulate")

    # Measure the accuracy scores
    sklearn_score = accuracy_score(sklearn_y_pred, y_test)
    concrete_score = accuracy_score(concrete_y_pred, y_test)
    
    bitwidth = circuit.graph.maximum_integer_bit_width()

    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # pylint: disable-next=no-member
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # Plot the decision boundaries.
    # For that, a color is assigned to each point in the mesh, which is obtained as a
    # cartesian product of [x_min, x_max] with [y_min, y_max].
    if hasattr(sklearn_model, "decision_function"):
        sklearn_Z = sklearn_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        concrete_Z = concrete_model.decision_function(
            np.c_[xx.ravel(), yy.ravel()],
            fhe="simulate",
        )
    else:
        sklearn_Z = sklearn_model.predict_proba(
            np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        )[:, 1]
        concrete_Z = concrete_model.predict_proba(
            np.c_[xx.ravel(), yy.ravel()],
            fhe="simulate",
        )[:, 1]

    for _, (ax, framework, score, Z) in enumerate(
        zip(
            axs,
            ["scikit-learn", "Concrete ML"],
            [sklearn_score, concrete_score],
            [sklearn_Z, concrete_Z],
        )
    ):

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k")

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
            levels=[0.5],
            linewidths=2,
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.tick_params(axis='x', labelsize=font_size_text)
        ax.tick_params(axis='y', labelsize=font_size_text)

        ax.set_title(
            f"Logistic Regression ({framework}) with {name} data.", fontsize=font_size_text
        )

        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            f"{score*100:0.1f}%",
            size=font_size_text,
            horizontalalignment="right",
        )

        if bitwidth and framework == "Concrete ML":
            ax.text(
                xx.max() - 0.3,
                yy.min() + 1.0,
                f"bit-width={bitwidth}",
                size=font_size_text,
                horizontalalignment="right",
            )