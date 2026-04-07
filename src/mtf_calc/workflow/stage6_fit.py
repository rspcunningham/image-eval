from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks, savgol_filter


DEFAULT_HARMONIC_COUNT = 5


def _mtf_results(base_frequency_lp_per_mm: float, harmonic_amplitudes: list[float]) -> dict[str, Any]:
    harmonics: list[dict[str, float | int]] = []
    for index, amplitude in enumerate(harmonic_amplitudes):
        harmonic_order = 2 * index + 1
        ideal_amplitude = 2.0 / (np.pi * harmonic_order)
        mtf = abs(float(amplitude)) / ideal_amplitude
        harmonics.append({
            "harmonicOrder": harmonic_order,
            "frequencyLpPerMm": float(base_frequency_lp_per_mm) * harmonic_order,
            "measuredAmplitude": float(amplitude),
            "idealAmplitude": float(ideal_amplitude),
            "mtf": float(mtf),
        })
    return {
        "baseFrequencyLpPerMm": float(base_frequency_lp_per_mm),
        "firstHarmonicMtf": harmonics[0]["mtf"] if harmonics else None,
        "harmonics": harmonics,
    }


def _estimate_period_samples(y: np.ndarray) -> float:
    if y.size < 6:
        return max(float(y.size), 4.0)
    window = min(y.size - (1 - y.size % 2), 11)
    if window >= 5:
        smoothed = savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    else:
        smoothed = y
    gradient = np.abs(np.diff(smoothed))
    prominence = max(float(np.std(gradient)), 1e-6)
    peaks, _ = find_peaks(gradient, prominence=prominence)
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


def _design_matrix(x: np.ndarray, *, period: float, phase: float, harmonic_count: int) -> np.ndarray:
    omega = 2.0 * np.pi / period
    columns = [x, np.ones_like(x)]
    for index in range(harmonic_count):
        harmonic = 2 * index + 1
        columns.append(np.sin(harmonic * (omega * x + phase)))
    return np.column_stack(columns)


def _solve_linear_coeffs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    period: float,
    phase: float,
    harmonic_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    design = _design_matrix(x, period=period, phase=phase, harmonic_count=harmonic_count)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs
    residual = float(np.sum(np.square(fitted - y)))
    return coeffs, fitted, residual


def fit_profile(
    normalized_profile: list[float] | None,
    *,
    raw_profile: list[float],
    base_frequency_lp_per_mm: float,
    crop_left: int,
    crop_right: int,
    black_mean: float | None,
    contrast: float | None,
    harmonic_count: int = DEFAULT_HARMONIC_COUNT,
) -> dict[str, Any] | None:
    if normalized_profile is None or contrast is None or abs(float(contrast)) <= 1e-9:
        return None

    values = np.asarray(normalized_profile, dtype=np.float64)
    n = int(values.size)
    start = int(crop_left)
    stop = n - int(crop_right)
    if stop - start < max(8, harmonic_count * 2 + 2):
        return None

    x_fit = np.arange(start, stop, dtype=np.float64)
    y_fit = values[start:stop]
    period0 = np.clip(_estimate_period_samples(y_fit), 4.0, max(float(y_fit.size) * 1.5, 6.0))
    min_period = max(4.0, period0 * 0.55)
    max_period = min(max(float(y_fit.size) * 1.5, 8.0), period0 * 1.6)
    if max_period <= min_period:
        max_period = min_period + 1.0

    best_period = period0
    best_phase = 0.0
    best_coeffs = None
    best_residual = np.inf
    for period in np.linspace(min_period, max_period, 48):
        for phase in np.linspace(-np.pi, np.pi, 48, endpoint=False):
            coeffs, _, residual = _solve_linear_coeffs(
                x_fit,
                y_fit,
                period=float(period),
                phase=float(phase),
                harmonic_count=harmonic_count,
            )
            if residual < best_residual:
                best_residual = residual
                best_period = float(period)
                best_phase = float(phase)
                best_coeffs = coeffs

    objective = lambda params: _solve_linear_coeffs(
        x_fit,
        y_fit,
        period=float(params[0]),
        phase=float(params[1]),
        harmonic_count=harmonic_count,
    )[2]
    optimized = minimize(
        objective,
        x0=np.array([best_period, best_phase], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(min_period, max_period), (-np.pi, np.pi)],
        options={"maxiter": 300},
    )
    period_fit = float(optimized.x[0]) if optimized.success else best_period
    phase_fit = float(optimized.x[1]) if optimized.success else best_phase
    coeffs, fitted_segment, residual = _solve_linear_coeffs(
        x_fit,
        y_fit,
        period=period_fit,
        phase=phase_fit,
        harmonic_count=harmonic_count,
    )
    if best_coeffs is not None and residual > best_residual:
        period_fit = best_period
        phase_fit = best_phase
        coeffs = best_coeffs
        fitted_segment = _design_matrix(x_fit, period=period_fit, phase=phase_fit, harmonic_count=harmonic_count) @ coeffs
        residual = best_residual

    x_full = np.arange(n, dtype=np.float64)
    normalized_fit = _design_matrix(x_full, period=period_fit, phase=phase_fit, harmonic_count=harmonic_count) @ coeffs
    raw_fit = normalized_fit * float(contrast) + float(black_mean)
    residuals = fitted_segment - y_fit
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    harmonic_amplitudes = [float(value) for value in coeffs[2:]]

    return {
        "harmonicCount": harmonic_count,
        "crop": {"left": start, "right": int(crop_right)},
        "fitStart": start,
        "fitStop": stop,
        "periodSamples": period_fit,
        "phaseRad": phase_fit,
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "harmonicAmplitudes": harmonic_amplitudes,
        "rmse": rmse,
        "normalizedFitProfile": normalized_fit.astype(np.float64).tolist(),
        "rawFitProfile": raw_fit.astype(np.float64).tolist(),
        "mtf": _mtf_results(base_frequency_lp_per_mm, harmonic_amplitudes),
        "success": bool(optimized.success or best_coeffs is not None),
        "status": int(getattr(optimized, "status", 0)),
        "message": str(getattr(optimized, "message", "grid search")),
    }
