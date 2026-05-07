from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from image_eval.square_wave_fit import FittedBarROIProfile, SquareWaveFit

if TYPE_CHECKING:
    from image_eval.mtf_results import MTFResult


def save_square_wave_fit_plot(
    profile: np.ndarray,
    fit: SquareWaveFit,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    figure = plot_square_wave_fit(profile, fit, title=title)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_bar_roi_fit_plot(
    fitted_profile: FittedBarROIProfile,
    output_path: Path,
    *,
    mtf_value: float | None = None,
) -> None:
    roi = fitted_profile.roi_profile
    save_square_wave_fit_plot(
        roi.profile,
        fitted_profile.fit,
        output_path,
        title=bar_roi_fit_plot_title(fitted_profile, mtf_value=mtf_value),
    )


def bar_roi_fit_plot_title(
    fitted_profile: FittedBarROIProfile,
    *,
    mtf_value: float | None = None,
) -> str:
    roi = fitted_profile.roi_profile
    title = (
        f"G{roi.group} E{roi.element} {roi.orientation} "
        f"({roi.frequency_lp_per_mm:.3g} cycles/mm)"
    )
    if mtf_value is not None:
        title = f"{title} - MTF {mtf_value:.4g}"
    return title


def save_mtf_curve_plot(results: Sequence[MTFResult], output_path: Path) -> None:
    figure = plot_mtf_curves(results)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_square_wave_fit(
    profile: np.ndarray,
    fit: SquareWaveFit,
    *,
    title: str | None = None,
) -> Figure:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError(f"profile is {profile.ndim}D; expected a 1D array")
    if len(profile) != len(fit.fitted_profile):
        raise ValueError("profile and fitted profile must have the same length")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    sample_positions = range(len(fit.fitted_profile))
    axis.plot(sample_positions, profile, color="#2f6fbb", linewidth=1.8, label="Line profile")
    axis.plot(
        sample_positions,
        fit.fitted_profile,
        color="#d24b2a",
        linewidth=2.0,
        label="Fitted square wave + baseline",
    )
    axis.set_xlabel("Profile sample")
    axis.set_ylabel("Normalized intensity")
    axis.set_title(title or "Line Profile Square-Wave Fit")
    axis.grid(True, color="#dddddd", linewidth=0.7)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def plot_mtf_curves(results: Sequence[MTFResult]) -> Figure:
    if not results:
        raise ValueError("cannot plot MTF curves without results")

    sorted_results = sorted(results, key=lambda result: result.frequency_lp_per_mm)
    frequencies = np.array(
        [result.frequency_lp_per_mm for result in sorted_results],
        dtype=np.float64,
    )
    x_mtf = _series_with_nan([result.x_mtf for result in sorted_results])
    y_mtf = _series_with_nan([result.y_mtf for result in sorted_results])
    average_mtf = np.array([result.average_mtf for result in sorted_results], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(frequencies, x_mtf, marker="o", linewidth=1.8, label="X MTF")
    axis.plot(frequencies, y_mtf, marker="s", linewidth=1.8, label="Y MTF")
    axis.plot(frequencies, average_mtf, marker="^", linewidth=2.2, label="Average MTF")
    axis.set_xlabel("Spatial frequency (cycles/mm)")
    axis.set_ylabel("MTF")
    axis.set_title("MTF by USAF 1951 Spatial Frequency")
    axis.grid(True, color="#dddddd", linewidth=0.7)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def _series_with_nan(values: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if value is None else value for value in values], dtype=np.float64)
