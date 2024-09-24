import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from concrete.ml.quantization import QuantizedModule


def plot_decision_boundary(
    model, X, y, title, display_points=True, fhe="disable", use_inference=False
):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if isinstance(model, QuantizedModule):
        Z = model.forward(grid, fhe=fhe)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

    else:
        #        model.eval()

        with torch.no_grad():
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
            if use_inference:
                Z = model.inference(grid_tensor)
            else:
                Z = model.forward(grid_tensor)
            _, Z = torch.max(Z, 1)
            Z = Z.numpy().reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    if display_points:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors="k")

    plt.title(title)
    plt.show()
