"""Train a real RBF SVM and visualize its 2D boundary and decision surface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils.data_generator import generate_ring_dataset
from utils.svm_utils import compute_decision_grid, make_decision_grid, train_svm


def plot_rbf_decision_surface() -> None:
    """Create 2D and 3D plots for an actual sklearn RBF SVM."""
    X, y = generate_ring_dataset(n_inner=35, n_outer=45, noise=0.08, random_seed=7)
    model = train_svm(X, y, kernel="rbf", C=10.0, gamma=1.0)

    xx, yy, grid_points = make_decision_grid((-3, 3), (-3, 3), resolution=160)
    zz = compute_decision_grid(model, xx, yy, grid_points)
    point_scores = model.decision_function(X)
    support_vectors = model.support_vectors_
    support_scores = model.decision_function(support_vectors)

    fig = plt.figure(figsize=(14, 6))
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    inner = y == 0
    outer = y == 1
    ax_2d.scatter(X[inner, 0], X[inner, 1], c="#2f80ed", label="Inner class", edgecolors="white")
    ax_2d.scatter(X[outer, 0], X[outer, 1], c="#eb5757", label="Outer class", edgecolors="white")
    ax_2d.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="Support vectors",
    )
    contour = ax_2d.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["gray", "#f2c94c", "gray"])
    ax_2d.clabel(contour, fmt={-1: "f=-1", 0: "f=0", 1: "f=+1"}, inline=True)
    ax_2d.set_title("2D dataset with RBF SVM decision boundary")
    ax_2d.set_xlabel("x")
    ax_2d.set_ylabel("y")
    ax_2d.set_aspect("equal", adjustable="box")
    ax_2d.legend(loc="upper right")

    surface = ax_3d.plot_surface(xx, yy, zz, cmap="coolwarm", alpha=0.78, linewidth=0, antialiased=True)
    ax_3d.contour(xx, yy, zz, levels=[0], zdir="z", offset=0, colors=["#f2c94c"], linewidths=3)
    ax_3d.scatter(X[inner, 0], X[inner, 1], point_scores[inner], c="#2f80ed", edgecolors="white")
    ax_3d.scatter(X[outer, 0], X[outer, 1], point_scores[outer], c="#eb5757", edgecolors="white")
    ax_3d.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        support_scores,
        s=95,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
    )
    ax_3d.set_title("3D decision function surface z = f(x, y)")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("decision score")
    fig.colorbar(surface, ax=ax_3d, shrink=0.65, pad=0.12, label="f(x, y)")

    fig.suptitle(
        "Real RBF SVM: this surface visualizes the decision function, not a literal 3D RBF feature map.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_rbf_decision_surface()
