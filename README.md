# SVM Kernel Trick 3D Interactive Demo

Educational demos for understanding how Support Vector Machines can learn nonlinear boundaries.

## Project Overview

This project has three independent parts:

1. A Manim animation that explains a simple feature mapping from 2D to 3D.
2. A real sklearn RBF SVM visualization of the decision function.
3. A Streamlit and Plotly app for exploring kernels, `C`, `gamma`, noise, and support vectors.

## Educational Story

The dataset has blue points near the origin and red points in an outer ring. In the original 2D plane, a straight line cannot separate the classes. A feature map can lift points into a higher-dimensional space where a linear separator may exist. Kernel SVMs use this idea to learn nonlinear decision boundaries in the original input space.

## Phase 1: Manim Kernel Trick Animation

`phase1_manim_kernel_trick.py` uses the teaching-friendly map:

```text
phi(x, y) = (x, y, x^2 + y^2)
```

Blue inner points stay low, red outer points lift higher, and a horizontal plane separates them in 3D. Projecting that plane back to 2D gives a circular decision boundary.

## Phase 2: Real RBF SVM Decision Surface

`phase2_rbf_decision_surface.py` trains an actual `sklearn.svm.SVC` with `kernel="rbf"`. It plots:

- The 2D data with `f(x, y) = 0` decision boundary.
- Margin contours `f(x, y) = -1` and `f(x, y) = +1`.
- A 3D surface where height is the decision function value.
- Support vectors highlighted with ring markers.

## Phase 3: Interactive Streamlit Demo

`phase3_streamlit_app.py` lets students adjust:

- Kernel: `linear`, `poly`, `rbf`, `sigmoid`
- `C`
- `gamma`
- polynomial degree
- dataset noise
- number of points
- random seed
- grid resolution

The 2D boundary and 3D decision surface update together.

## Installation

```bash
pip install -r requirements.txt
```

Manim may require extra system dependencies depending on your platform. See the Manim Community installation guide if rendering fails.

On Windows, Python versions with prebuilt wheels for Manim dependencies are easiest. If `pip install -r requirements.txt` fails while building `moderngl` or `glcontext`, use Python 3.11 or 3.12, or install Microsoft C++ Build Tools.

## Run Commands

```bash
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
python phase2_rbf_decision_surface.py
streamlit run phase3_streamlit_app.py
```

## Important Mathematical Note

The mapping `z = x^2 + y^2` is used as a visual and educational feature mapping to explain why nonlinear data can become linearly separable in a higher-dimensional feature space.

A real RBF kernel does not explicitly map data to only 3D. It corresponds to a high-dimensional or infinite-dimensional feature space. The RBF decision surface shown in Phase 2 and Phase 3 visualizes the decision function `f(x, y)`, not the full feature space itself.

## Teaching Suggestions

Start with the Manim animation to build intuition. Then run the RBF surface script to show a real trained model. Finally, use the Streamlit app in class and ask students to compare small versus large `gamma`, and small versus large `C`.

Useful prompts:

- What happens to the boundary when `gamma` is very small?
- What happens when `gamma` is very large?
- How does increasing `C` change the model's tolerance for mistakes?
- Which points become support vectors, and why do they matter?
