# SVM and Manim Demo - Work Summary

This project implements an educational Support Vector Machine kernel trick demo with three parts:

1. A Manim 3D animation for the concept mapping `phi(x, y) = (x, y, x^2 + y^2)`.
2. A real sklearn RBF SVM decision-function visualization.
3. An interactive Streamlit and Plotly app for exploring kernel parameters.

## What Was Built

- `phase1_manim_kernel_trick.py`: Manim `ThreeDScene` showing circular 2D data lifted into 3D, separated by a hyperplane, then projected back as a circular boundary.
- `phase2_rbf_decision_surface.py`: sklearn `SVC(kernel="rbf")` script with 2D boundary, margins, support vectors, and a 3D decision-function surface.
- `phase3_streamlit_app.py`: interactive app with controls for kernel, `C`, `gamma`, polynomial degree, noise, point count, seed, and grid resolution.
- `utils/data_generator.py`: reusable circular/ring dataset generator and concept lift helper.
- `utils/svm_utils.py`: reusable SVM training and decision-grid helpers.
- `README.md`: setup, run commands, educational notes, and Manim environment caveat.
- `log.md`: implementation log and verification notes.

## Run Commands

```bash
pip install -r requirements.txt
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
python phase2_rbf_decision_surface.py
streamlit run phase3_streamlit_app.py
```

## Verification

Completed checks:

- Python syntax check passed.
- sklearn utility smoke test passed.
- Plotly 2D and 3D figure construction passed.
- Streamlit app launched locally and responded with HTTP 200.

## Important Note

The Manim animation uses `z = x^2 + y^2` as a teaching-friendly feature map. A real RBF kernel is not simply a 3D mapping; the Phase 2 and Phase 3 plots show the learned decision function `f(x, y)`.

On Windows with Python 3.14, Manim dependency installation may require Microsoft C++ Build Tools. Python 3.11 or 3.12 is recommended for easier Manim setup.
