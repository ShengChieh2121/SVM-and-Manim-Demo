# Implementation Log

## Summary

Implemented the `design.md` project as a complete SVM Kernel Trick educational demo. The project now includes a Manim concept animation, a real sklearn RBF SVM decision-surface script, and an interactive Streamlit/Plotly app.

## Files Created

- `requirements.txt`
- `README.md`
- `phase1_manim_kernel_trick.py`
- `phase2_rbf_decision_surface.py`
- `phase3_streamlit_app.py`
- `utils/__init__.py`
- `utils/data_generator.py`
- `utils/svm_utils.py`
- `assets/.gitkeep`
- `outputs/.gitkeep`

## Utility Layer

Created reusable helpers in `utils/`:

- `generate_ring_dataset(...)`: creates blue inner-class points and red outer-ring points.
- `concept_lift(...)`: maps `(x, y)` to `(x, y, x^2 + y^2)` for the teaching animation.
- `train_svm(...)`: trains an sklearn `SVC`.
- `make_decision_grid(...)`: builds a 2D mesh grid.
- `compute_decision_surface(...)`: computes decision scores.
- `compute_decision_grid(...)`: reshapes decision scores back to grid form.

## Phase 1

Implemented `phase1_manim_kernel_trick.py`.

The scene class `SVMKernelTrick3D` shows:

- 2D circular data that is not linearly separable.
- The educational feature map `phi(x, y) = (x, y, x^2 + y^2)`.
- Points lifting into 3D.
- A translucent paraboloid `z = x^2 + y^2`.
- A horizontal separating hyperplane.
- The projected 2D circular decision boundary.
- A final summary distinguishing 3D linear separation from 2D nonlinear boundaries.

## Phase 2

Implemented `phase2_rbf_decision_surface.py`.

The script trains a real sklearn `SVC(kernel="rbf")` and plots:

- The 2D dataset.
- The decision boundary `f(x, y) = 0`.
- Margin contours `f(x, y) = -1` and `f(x, y) = +1`.
- Support vectors.
- A 3D matplotlib decision-function surface.

Important distinction preserved: the RBF plot visualizes the decision function, not a literal 3D RBF feature map.

## Phase 3

Implemented `phase3_streamlit_app.py`.

The app includes:

- Kernel selector: `linear`, `poly`, `rbf`, `sigmoid`.
- Controls for `C`, `gamma`, degree, noise, number of points, random seed, and grid resolution.
- 2D Plotly decision-boundary visualization.
- 3D Plotly decision-function surface.
- Support-vector metrics and table.
- Teaching notes that respond to `C` and `gamma`.
- Streamlit caching for dataset/model-grid calculations.

## README

Created `README.md` with:

- Project overview.
- Educational story.
- Phase-by-phase explanation.
- Installation instructions.
- Run commands.
- Mathematical note about the difference between the concept mapping and real RBF kernels.
- Teaching suggestions.

## Verification

Completed:

- Python syntax check for all `.py` files: passed.
- Utility smoke test with sklearn SVM training: passed after installing non-Manim dependencies.
- Plotly figure construction smoke test for both 2D and 3D figures: passed.
- Streamlit app launched locally and responded with HTTP `200`.

Local app URL:

```text
http://localhost:8501
```

## Environment Caveat

`pip install -r requirements.txt` failed on this machine while installing Manim because `moderngl` and `glcontext` needed Microsoft C++ Build Tools under Python 3.14.

Workaround options:

- Use Python 3.11 or 3.12 for better prebuilt wheel availability.
- Install Microsoft C++ Build Tools.
- Install and test the non-Manim dependencies separately when only running Phase 2 and Phase 3.
