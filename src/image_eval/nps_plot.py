from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

if TYPE_CHECKING:
    from image_eval.nps_results import NPSResult, NPSSpectrum


def save_nps_curve_plot(
    results: Sequence[NPSResult],
    output_path: Path,
    *,
    frequency_unit: str,
) -> None:
    figure = plot_nps_curves(results, frequency_unit=frequency_unit)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_nps_curves(results: Sequence[NPSResult], *, frequency_unit: str) -> Figure:
    if not results:
        raise ValueError("cannot plot NPS curves without results")

    frequencies = np.array([result.frequency for result in results], dtype=np.float64)
    black_nps = _series_with_nan([result.black_nps for result in results])
    white_nps = _series_with_nan([result.white_nps for result in results])

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(frequencies, black_nps, marker="o", linewidth=1.8, label="Black NPS")
    axis.plot(frequencies, white_nps, marker="s", linewidth=1.8, label="White NPS")
    axis.set_xlabel(f"Spatial frequency ({frequency_unit})")
    axis.set_ylabel("NPS")
    axis.set_title("Noise Power Spectrum")
    axis.grid(True, color="#dddddd", linewidth=0.7)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def save_nps_spectrum_plot(
    spectrum: NPSSpectrum,
    output_path: Path,
    *,
    frequency_unit: str,
) -> None:
    figure = plot_nps_spectrum(spectrum, frequency_unit=frequency_unit)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_nps_spectrum(spectrum: NPSSpectrum, *, frequency_unit: str) -> Figure:
    power = np.asarray(spectrum.power_spectrum, dtype=np.float64)
    if power.ndim != 2:
        raise ValueError(f"power spectrum is {power.ndim}D; expected a 2D array")

    positive_power = power[power > 0]
    floor = float(np.min(positive_power)) if positive_power.size else np.finfo(np.float64).tiny
    log_power = np.log10(np.maximum(power, floor))

    frequency_x = np.asarray(spectrum.frequency_x, dtype=np.float64)
    frequency_y = np.asarray(spectrum.frequency_y, dtype=np.float64)
    x_min, x_max = _axis_extent(frequency_x)
    y_min, y_max = _axis_extent(frequency_y)

    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    image = axis.imshow(
        log_power,
        cmap="viridis",
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )
    axis.set_xlabel(f"Frequency x ({frequency_unit})")
    axis.set_ylabel(f"Frequency y ({frequency_unit})")
    axis.set_title(f"{spectrum.roi_name.capitalize()} ROI 2D NPS")
    figure.colorbar(image, ax=axis, label="log10 NPS")
    figure.tight_layout()
    return figure


def _series_with_nan(values: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if value is None else value for value in values], dtype=np.float64)


def _axis_extent(axis: np.ndarray) -> tuple[float, float]:
    if axis.ndim != 1:
        raise ValueError(f"frequency axis is {axis.ndim}D; expected a 1D array")
    if axis.size == 0:
        raise ValueError("frequency axis must not be empty")
    if axis.size == 1:
        center = float(axis[0])
        return center - 0.5, center + 0.5

    step = float(np.mean(np.diff(axis)))
    return float(axis[0] - step / 2.0), float(axis[-1] + step / 2.0)
