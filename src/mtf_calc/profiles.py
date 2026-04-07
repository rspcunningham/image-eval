# pyright: reportAny=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

import numpy as np
from collections.abc import Mapping
from numpy.typing import NDArray
from typing import Protocol, cast
from scipy.optimize import minimize as _scipy_minimize
from scipy.signal import find_peaks as _scipy_find_peaks, savgol_filter as _scipy_savgol_filter

from mtf_calc.models import Dim, FitResult, NormRegion, Profile, Roi


class _OptimizeResult(Protocol):
    x: NDArray[np.float64]
    success: bool


def get_norm(
    raw_image: NDArray[np.float32],
    norm_roi_dict: dict[NormRegion, Roi],
) -> tuple[float, float]:
    def roi_mean(roi: Roi) -> float:
        x0 = int(roi.top_left.x)
        x1 = int(roi.top_right.x)
        y0 = int(roi.top_left.y)
        y1 = int(roi.bottom_left.y)

        return float(raw_image[y0:y1, x0:x1].mean())

    return roi_mean(norm_roi_dict[0]), roi_mean(norm_roi_dict[1])

def extract(
    raw_image: NDArray[np.float32],
    *,
    bar_roi: Roi,
    norm_rois: Mapping[NormRegion, Roi],
    dim: Dim,
) -> Profile:

    x0 = int(bar_roi.top_left.x)
    x1 = int(bar_roi.top_right.x)
    y0 = int(bar_roi.top_left.y)
    y1 = int(bar_roi.bottom_left.y)

    bar_values = raw_image[y0:y1, x0:x1]
    raw_values = bar_values.mean(axis=1 if dim == "Y" else 0)

    black_norm, white_norm = get_norm(raw_image, dict(norm_rois))
    range_norm = white_norm - black_norm
    norm_values = (raw_values - black_norm) / range_norm

    return Profile(
        raw_values=[float(value) for value in raw_values],
        norm_values=[float(value) for value in norm_values],
    )


def fit(
    profile: Profile,
    *,
    norm_rois: Mapping[NormRegion, Roi],
    n_harmonics: int,
) -> FitResult:
    _ = norm_rois
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1.")

    values: NDArray[np.float64] = np.asarray(profile.norm_values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        raise ValueError("Profile is empty.")
    if n < max(8, n_harmonics * 2 + 2):
        raise ValueError("Profile has too few samples to fit requested harmonics.")
    if not np.isfinite(values).all():
        raise ValueError("Profile contains non-finite values.")

    def estimate_period(y: NDArray[np.float64]) -> float:
        if y.size < 6:
            return max(float(y.size), 4.0)

        window = min(y.size - (1 - y.size % 2), 11)
        if window >= 5:
            smoothed = cast(
                NDArray[np.float64],
                _scipy_savgol_filter(y, window_length=window, polyorder=2, mode="interp"),
            )
        else:
            smoothed = y

        gradient = np.abs(np.diff(smoothed))
        prominence = max(float(np.std(gradient)), 1e-6)
        peaks, _ = cast(
            tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]],
            _scipy_find_peaks(gradient, prominence=prominence),
        )

        if peaks.size >= 2:
            return float(max(2.0 * np.median(np.diff(peaks)), 4.0))

        x = np.arange(y.size, dtype=np.float64)
        trend = np.polyval(np.polyfit(x, y, deg=1), x)
        centered = y - trend
        autocorr = np.correlate(centered, centered, mode="full")[centered.size - 1:]
        search = autocorr[2:max(3, centered.size // 2)]
        if search.size == 0 or np.allclose(search, 0):
            return max(centered.size / 3.0, 4.0)
        return float(max(np.argmax(search) + 2, 4.0))

    def design_matrix(
        x: NDArray[np.float64],
        *,
        period: float,
        phase: float,
        harmonic_count: int,
    ) -> NDArray[np.float64]:
        omega = 2.0 * np.pi / period
        columns = [x, np.ones_like(x)]
        for index in range(harmonic_count):
            harmonic = 2 * index + 1
            columns.append(np.sin(harmonic * (omega * x + phase)))
        return np.column_stack(columns)

    def solve_linear_coeffs(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        period: float,
        phase: float,
        harmonic_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        design = design_matrix(x, period=period, phase=phase, harmonic_count=harmonic_count)
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        coeffs = cast(NDArray[np.float64], coeffs)
        fitted = design @ coeffs
        residual = float(np.sum(np.square(fitted - y)))
        return coeffs, fitted, residual

    x = np.arange(n, dtype=np.float64)
    y: NDArray[np.float64] = values.astype(np.float64, copy=False)

    period0 = estimate_period(y)
    period0 = np.clip(period0, 4.0, max(float(y.size) * 1.5, 6.0))
    min_period = max(4.0, period0 * 0.55)
    max_period = min(max(float(y.size) * 1.5, 8.0), period0 * 1.6)
    if max_period <= min_period:
        max_period = min_period + 1.0

    best_period = period0
    best_phase = 0.0
    best_coeffs = np.zeros(2 + n_harmonics, dtype=np.float64)
    best_residual = np.inf

    for period in np.linspace(min_period, max_period, 48):
        for phase in np.linspace(-np.pi, np.pi, 48, endpoint=False):
            coeffs, _, residual = solve_linear_coeffs(
                x,
                y,
                period=float(period),
                phase=float(phase),
                harmonic_count=n_harmonics,
            )
            if residual < best_residual:
                best_residual = residual
                best_period = float(period)
                best_phase = float(phase)
                best_coeffs = coeffs

    def objective(params: NDArray[np.float64]) -> float:
        return solve_linear_coeffs(
            x,
            y,
            period=float(params[0]),
            phase=float(params[1]),
            harmonic_count=n_harmonics,
        )[2]

    optimized = cast(
        _OptimizeResult,
        _scipy_minimize(
        objective,
        x0=np.array([best_period, best_phase], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(min_period, max_period), (-np.pi, np.pi)],
        options={"maxiter": 300},
        ),
    )

    period_px = float(optimized.x[0]) if optimized.success else best_period
    phase_rad = float(optimized.x[1]) if optimized.success else best_phase
    coeffs, _, residual = solve_linear_coeffs(
        x,
        y,
        period=period_px,
        phase=phase_rad,
        harmonic_count=n_harmonics,
    )
    if residual > best_residual:
        period_px = best_period
        phase_rad = best_phase
        coeffs = best_coeffs

    harmonic_amplitudes = [float(value) for value in coeffs[2:]]

    return FitResult(
        period_px=period_px,
        phase_rad=phase_rad,
        harmonic_amplitudes=harmonic_amplitudes,
        slope=float(coeffs[0]),
        intercept=float(coeffs[1]),
    )
