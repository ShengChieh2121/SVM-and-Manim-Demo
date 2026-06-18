"""Interactive Streamlit and Plotly demo for SVM kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_generator import generate_ring_dataset
from utils.svm_utils import compute_decision_grid, make_decision_grid, train_svm


BLUE = "#2f80ed"
RED = "#eb5757"
YELLOW = "#f2c94c"


@st.cache_data(show_spinner=False)
def cached_dataset(n_points: int, noise: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_inner = max(10, int(round(n_points * 0.44)))
    n_outer = n_points - n_inner
    return generate_ring_dataset(n_inner=n_inner, n_outer=n_outer, noise=noise, random_seed=random_seed)


@st.cache_data(show_spinner=False)
def cached_decision_data(
    n_points: int,
    noise: float,
    random_seed: int,
    kernel: str,
    C: float,
    gamma: float,
    degree: int,
    resolution: int,
) -> dict[str, object]:
    X, y = cached_dataset(n_points, noise, random_seed)
    model_gamma: float | str = gamma if kernel in {"rbf", "poly", "sigmoid"} else "scale"
    model = train_svm(X, y, kernel=kernel, C=C, gamma=model_gamma, degree=degree)
    xx, yy, grid_points = make_decision_grid((-3, 3), (-3, 3), resolution=resolution)
    zz = compute_decision_grid(model, xx, yy, grid_points)
    point_scores = model.decision_function(X)
    predictions = model.predict(X)
    return {
        "X": X,
        "y": y,
        "model": model,
        "xx": xx,
        "yy": yy,
        "zz": zz,
        "point_scores": point_scores,
        "accuracy": float(np.mean(predictions == y)),
    }


def make_2d_plot(X: np.ndarray, y: np.ndarray, model, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> go.Figure:
    support = model.support_vectors_
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            contours=dict(start=-1, end=1, size=1, coloring="none", showlabels=True),
            line=dict(color="rgba(130,130,130,0.9)", width=1.5),
            showscale=False,
            name="Margins",
        )
    )
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            contours=dict(start=0, end=0, size=1, coloring="none"),
            line=dict(color=YELLOW, width=4),
            showscale=False,
            name="f(x,y)=0",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            mode="markers",
            marker=dict(color=BLUE, size=8, line=dict(color="white", width=1)),
            name="Inner class",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode="markers",
            marker=dict(color=RED, size=8, line=dict(color="white", width=1)),
            name="Outer class",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=support[:, 0],
            y=support[:, 1],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=15, line=dict(color="black", width=2)),
            name="Support vectors",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(title="x", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def make_3d_plot(
    X: np.ndarray,
    y: np.ndarray,
    model,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    point_scores: np.ndarray,
) -> go.Figure:
    support = model.support_vectors_
    support_scores = model.decision_function(support)
    zero_plane = np.zeros_like(zz)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale="RdBu",
            opacity=0.82,
            showscale=True,
            colorbar=dict(title="f(x,y)"),
            name="Decision surface",
        )
    )
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zero_plane,
            opacity=0.18,
            showscale=False,
            colorscale=[[0, YELLOW], [1, YELLOW]],
            name="z=0",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            z=point_scores[y == 0],
            mode="markers",
            marker=dict(color=BLUE, size=4, line=dict(color="white", width=1)),
            name="Inner class",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            z=point_scores[y == 1],
            mode="markers",
            marker=dict(color=RED, size=4, line=dict(color="white", width=1)),
            name="Outer class",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=support[:, 0],
            y=support[:, 1],
            z=support_scores,
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=7, line=dict(color="black", width=4)),
            name="Support vectors",
        )
    )
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=35, b=0),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="decision score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def teaching_notes(kernel: str, C: float, gamma: float) -> list[str]:
    notes = [
        "The z = x^2 + y^2 animation is a concept map. The RBF view here shows f(x, y), not the full RBF feature space.",
        f"Current kernel: {kernel}. The yellow contour marks f(x, y) = 0.",
    ]
    if kernel in {"rbf", "poly", "sigmoid"}:
        if gamma < 0.2:
            notes.append("Gamma is small: the boundary is smoother and each point has wider influence.")
        if gamma > 3:
            notes.append("Gamma is large: the boundary becomes very flexible and may overfit.")
    if C < 1:
        notes.append("C is small: the model allows more mistakes to keep a wider margin.")
    if C > 20:
        notes.append("C is large: the model tries harder to classify training data correctly.")
    return notes


def main() -> None:
    st.set_page_config(page_title="Interactive SVM Kernel Trick 3D Demo", layout="wide")
    st.title("Interactive SVM Kernel Trick 3D Demo")

    with st.sidebar:
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
        C = st.slider("C", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        gamma = 1.0
        if kernel in {"rbf", "poly", "sigmoid"}:
            gamma = st.slider("Gamma", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        degree = 3
        if kernel == "poly":
            degree = st.slider("Degree", min_value=2, max_value=6, value=3, step=1)
        noise = st.slider("Noise", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
        n_points = st.slider("Number of points", min_value=40, max_value=300, value=120, step=10)
        random_seed = st.number_input("Random seed", value=7, step=1)
        resolution = st.slider("Grid resolution", min_value=50, max_value=150, value=80, step=10)

    data = cached_decision_data(
        n_points=n_points,
        noise=noise,
        random_seed=int(random_seed),
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        resolution=resolution,
    )
    X = data["X"]
    y = data["y"]
    model = data["model"]
    xx = data["xx"]
    yy = data["yy"]
    zz = data["zz"]
    point_scores = data["point_scores"]

    with st.expander("Concept", expanded=True):
        st.write(
            "2D circular data cannot be separated by a straight line. Kernel methods let SVM learn nonlinear "
            "decision boundaries. RBF uses similarity to support vectors to form a flexible boundary."
        )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Support vectors", len(model.support_))
    metric_cols[1].metric("Training accuracy", f"{data['accuracy']:.1%}")
    metric_cols[2].metric("Kernel", kernel)
    metric_cols[3].metric("C", f"{C:.1f}")
    metric_cols[4].metric("Gamma", f"{gamma:.2f}" if kernel != "linear" else "scale")

    tab_2d, tab_3d, tab_data, tab_notes = st.tabs(
        ["2D Decision Boundary", "3D Decision Function Surface", "Support Vectors", "Teaching Notes"]
    )
    with tab_2d:
        st.plotly_chart(make_2d_plot(X, y, model, xx, yy, zz), use_container_width=True)
    with tab_3d:
        st.plotly_chart(make_3d_plot(X, y, model, xx, yy, zz, point_scores), use_container_width=True)
    with tab_data:
        support_df = pd.DataFrame(model.support_vectors_, columns=["x", "y"])
        support_df["decision_score"] = model.decision_function(model.support_vectors_)
        st.dataframe(support_df, use_container_width=True)
    with tab_notes:
        for note in teaching_notes(kernel, C, gamma):
            st.info(note)


if __name__ == "__main__":
    main()
