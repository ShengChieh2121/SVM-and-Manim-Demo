"""SVM helpers shared by the static and interactive demos."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: float | str = 1.0,
    degree: int = 3,
) -> SVC:
    """Train an sklearn SVC with parameters exposed in the demos."""
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    return model.fit(X, y)


def make_decision_grid(
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a mesh grid and its flattened point representation."""
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, grid_points


def compute_decision_surface(model: SVC, grid_points: np.ndarray) -> np.ndarray:
    """Compute decision scores for flattened grid points."""
    return model.decision_function(grid_points)


def compute_decision_grid(
    model: SVC,
    xx: np.ndarray,
    yy: np.ndarray,
    grid_points: np.ndarray,
) -> np.ndarray:
    """Compute decision scores and reshape them to the supplied mesh grid."""
    scores = compute_decision_surface(model, grid_points)
    return scores.reshape(xx.shape)
