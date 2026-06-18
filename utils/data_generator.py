"""Dataset helpers for the SVM kernel trick demos."""

from __future__ import annotations

import numpy as np


def _sample_annulus(
    rng: np.random.Generator,
    n_points: int,
    radius_range: tuple[float, float],
) -> np.ndarray:
    """Sample points uniformly by area inside an annulus."""
    r_min, r_max = radius_range
    radii = np.sqrt(rng.uniform(r_min**2, r_max**2, n_points))
    angles = rng.uniform(0.0, 2.0 * np.pi, n_points)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def generate_ring_dataset(
    n_inner: int = 35,
    n_outer: int = 45,
    inner_radius_range: tuple[float, float] = (0.0, 1.0),
    outer_radius_range: tuple[float, float] = (1.6, 2.5),
    noise: float = 0.08,
    random_seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate blue inner points and red outer-ring points.

    Labels use 0 for the inner class and 1 for the outer class so the same
    dataset can drive the concept animation, sklearn script, and Streamlit app.
    """
    rng = np.random.default_rng(random_seed)
    inner = _sample_annulus(rng, n_inner, inner_radius_range)
    outer = _sample_annulus(rng, n_outer, outer_radius_range)

    if noise > 0:
        inner += rng.normal(0.0, noise, inner.shape)
        outer += rng.normal(0.0, noise, outer.shape)

    X = np.vstack((inner, outer))
    y = np.concatenate((np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)))
    return X, y


def concept_lift(X: np.ndarray) -> np.ndarray:
    """Map (x, y) to the teaching-friendly 3D feature map (x, y, x^2 + y^2)."""
    z = np.sum(X**2, axis=1)
    return np.column_stack((X, z))
