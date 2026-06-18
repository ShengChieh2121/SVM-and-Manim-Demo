"""Manim animation for the teaching-friendly SVM kernel trick intuition."""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN,
    LEFT,
    MathTex,
    ORIGIN,
    ParametricFunction,
    RED,
    RIGHT,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    VGroup,
    YELLOW,
    config,
    Dot3D,
)

from utils.data_generator import concept_lift, generate_ring_dataset


class SVMKernelTrick3D(ThreeDScene):
    """Show how z = x^2 + y^2 can make circular data linearly separable."""

    def construct(self) -> None:
        config.background_color = "#111318"
        X, y = generate_ring_dataset(noise=0.0, random_seed=7)
        lifted = concept_lift(X)
        threshold = 2.05

        title = Text("SVM Kernel Trick: From 2D to 3D", font_size=42)
        subtitle = Text("Nonlinear in 2D, linear in feature space.", font_size=26)
        subtitle.next_to(title, DOWN)
        opening = VGroup(title, subtitle)
        self.play(FadeIn(opening))
        self.wait(1.2)
        self.play(FadeOut(opening))

        axes = ThreeDAxes(
            x_range=(-3, 3, 1),
            y_range=(-3, 3, 1),
            z_range=(0, 7, 1),
            x_length=6,
            y_length=6,
            z_length=4.5,
        )
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.play(FadeIn(axes))

        flat_points = self._points_for(X, y, axes, z_values=np.zeros(len(X)))
        note_2d = Text("No straight line can separate them in 2D.", font_size=26)
        note_2d.to_corner(LEFT + DOWN)
        self.add_fixed_in_frame_mobjects(note_2d)
        self.play(FadeIn(flat_points), FadeIn(note_2d))
        self.wait(1.5)

        formula = MathTex(r"\phi(x,y)=(x,y,x^2+y^2)", font_size=42)
        formula.to_corner(RIGHT + DOWN)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(FadeIn(formula))
        self.wait(1.0)

        lifted_points = self._points_for(X, y, axes, z_values=lifted[:, 2])
        self.play(Transform(flat_points, lifted_points), run_time=2.5)
        self.wait(0.8)

        surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=(-2.55, 2.55),
            v_range=(-2.55, 2.55),
            resolution=(24, 24),
            fill_opacity=0.22,
            checkerboard_colors=[GREEN, GREEN],
            stroke_opacity=0.15,
        )
        self.play(FadeIn(surface), run_time=1.5)

        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, threshold),
            u_range=(-2.7, 2.7),
            v_range=(-2.7, 2.7),
            resolution=(2, 2),
            fill_opacity=0.35,
            checkerboard_colors=[YELLOW, YELLOW],
            stroke_opacity=0.25,
        )
        hyperplane_label = Text("Hyperplane in feature space", font_size=24)
        hyperplane_label.to_corner(RIGHT + DOWN)
        self.add_fixed_in_frame_mobjects(hyperplane_label)
        self.play(FadeOut(formula), FadeIn(hyperplane), FadeIn(hyperplane_label))
        self.wait(1.2)

        projection = MathTex(r"z=c,\ z=x^2+y^2 \Rightarrow x^2+y^2=c", font_size=36)
        projection.to_corner(RIGHT + DOWN)
        self.add_fixed_in_frame_mobjects(projection)
        circle = ParametricFunction(
            lambda t: axes.c2p(np.sqrt(threshold) * np.cos(t), np.sqrt(threshold) * np.sin(t), 0),
            t_range=(0, 2 * np.pi),
            color=YELLOW,
            stroke_width=5,
        )
        self.play(FadeOut(hyperplane_label), FadeIn(projection), FadeIn(circle))
        self.wait(1.0)

        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(4.0)
        self.stop_ambient_camera_rotation()

        summary = VGroup(
            Text("In 3D: linear hyperplane", font_size=28),
            Text("In 2D: nonlinear decision boundary", font_size=28),
            Text("This is the intuition behind the kernel trick.", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT)
        summary.to_corner(LEFT + DOWN)
        self.add_fixed_in_frame_mobjects(summary)
        self.play(FadeOut(note_2d), FadeOut(projection), FadeIn(summary))
        self.wait(2.0)

    @staticmethod
    def _points_for(
        X: np.ndarray,
        y: np.ndarray,
        axes: ThreeDAxes,
        z_values: np.ndarray,
    ) -> VGroup:
        points = VGroup()
        for (x_coord, y_coord), label, z_coord in zip(X, y, z_values):
            color = BLUE if label == 0 else RED
            points.add(Dot3D(point=axes.c2p(x_coord, y_coord, z_coord), radius=0.045, color=color))
        return points
