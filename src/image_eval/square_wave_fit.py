from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from image_eval.mtf_profiles import BarROIProfile


SQUARE_WAVE_TERMS = 3
CYCLE_SEARCH_MIN = 2.0
CYCLE_SEARCH_MAX = 4.0
CYCLE_SEARCH_STEPS = 401
# Fraction of each profile (split across both ends) down-weighted by a Tukey taper, so
# ROI margins and edge halos beyond the bars carry little weight in the fit.
TAPER_FRACTION = 0.5
# Free-period fits used to calibrate pixels per millimetre must at least reach this
# fundamental MTF; poorly resolved elements would otherwise fit noise at any period.
CALIBRATION_MIN_MTF = 0.3
# Odd harmonics are fit only below this frequency (cycles/mm); the fundamental is always
# fit. Above it, harmonics of the bar pattern fall past the optics' passband and the
# sampling limit, so their coefficients would only fit noise.
MAX_HARMONIC_CYCLES_PER_MM = 350.0


@dataclass(frozen=True)
class SquareWaveFit:
    terms: int
    cycles: float
    harmonics: np.ndarray
    offset: float
    baseline_slope: float
    sine_coefficients: np.ndarray
    cosine_coefficients: np.ndarray
    fitted_profile: np.ndarray
    residuals: np.ndarray
    residual_rms: float

    @property
    def fundamental_amplitude(self) -> float:
        return float(np.hypot(self.sine_coefficients[0], self.cosine_coefficients[0]))


@dataclass(frozen=True)
class FittedBarROIProfile:
    roi_profile: BarROIProfile
    fit: SquareWaveFit


def fit_bar_roi_profiles(roi_profiles: Sequence[BarROIProfile]) -> list[FittedBarROIProfile]:
    """Fit every ROI at the period implied by its known frequency.

    Pixels per millimetre are calibrated once from free-period fits of well-resolved
    elements, then each ROI is refit with its cycle count fixed, so poorly resolved
    elements cannot pick whichever period best matches their noise.
    """
    harmonics = [harmonics_for_frequency(roi.frequency_lp_per_mm) for roi in roi_profiles]
    free_fits = [
        fit_square_wave_profile(roi.profile, harmonics=roi_harmonics)
        for roi, roi_harmonics in zip(roi_profiles, harmonics, strict=True)
    ]
    pixels_per_mm = calibrate_pixels_per_mm(roi_profiles, free_fits)
    return [
        FittedBarROIProfile(
            roi_profile=roi,
            fit=fit_square_wave_profile_at_cycles(
                roi.profile,
                cycles=roi.frequency_lp_per_mm * len(roi.profile) / pixels_per_mm,
                harmonics=roi_harmonics,
            ),
        )
        for roi, roi_harmonics in zip(roi_profiles, harmonics, strict=True)
    ]


def harmonics_for_frequency(fundamental_cycles_per_mm: float) -> np.ndarray:
    """Odd harmonics to fit: the fundamental, plus each harmonic below the frequency cap."""
    harmonics = _odd_harmonics()
    keep = (harmonics == 1) | (harmonics * fundamental_cycles_per_mm < MAX_HARMONIC_CYCLES_PER_MM)
    return harmonics[keep]


def calibrate_pixels_per_mm(
    roi_profiles: Sequence[BarROIProfile],
    free_fits: Sequence[SquareWaveFit],
) -> float:
    if not roi_profiles:
        raise ValueError("cannot calibrate pixels per millimetre without ROI profiles")
    candidates = [
        roi_profile.frequency_lp_per_mm * len(roi_profile.profile) / fit.cycles
        for roi_profile, fit in zip(roi_profiles, free_fits, strict=True)
        if CYCLE_SEARCH_MIN < fit.cycles < CYCLE_SEARCH_MAX
        and fit.fundamental_amplitude * np.pi / 2.0 >= CALIBRATION_MIN_MTF
    ]
    if not candidates:
        candidates = [
            roi_profile.frequency_lp_per_mm * len(roi_profile.profile) / fit.cycles
            for roi_profile, fit in zip(roi_profiles, free_fits, strict=True)
        ]
    return float(np.median(candidates))


def fit_square_wave_profile_at_cycles(
    profile: np.ndarray,
    *,
    cycles: float,
    harmonics: np.ndarray | None = None,
) -> SquareWaveFit:
    return _fit_square_wave_profile_at_cycles(
        _as_1d_finite_profile(profile),
        harmonics=_odd_harmonics() if harmonics is None else harmonics,
        cycles=cycles,
    )


def fit_square_wave_profile(
    profile: np.ndarray,
    *,
    harmonics: np.ndarray | None = None,
) -> SquareWaveFit:
    profile = _as_1d_finite_profile(profile)
    return _best_cycle_fit(
        profile,
        harmonics=_odd_harmonics() if harmonics is None else harmonics,
    )


def _best_cycle_fit(
    profile: np.ndarray,
    *,
    harmonics: np.ndarray,
) -> SquareWaveFit:
    candidate_cycles = np.linspace(CYCLE_SEARCH_MIN, CYCLE_SEARCH_MAX, CYCLE_SEARCH_STEPS)
    best_fit = _fit_square_wave_profile_at_cycles(
        profile,
        harmonics=harmonics,
        cycles=float(candidate_cycles[0]),
    )
    for cycles in candidate_cycles[1:]:
        fit = _fit_square_wave_profile_at_cycles(
            profile,
            harmonics=harmonics,
            cycles=float(cycles),
        )
        if fit.residual_rms < best_fit.residual_rms:
            best_fit = fit

    return best_fit


def _fit_square_wave_profile_at_cycles(
    profile: np.ndarray,
    *,
    harmonics: np.ndarray,
    cycles: float,
) -> SquareWaveFit:
    design = _square_wave_design_matrix(len(profile), cycles=cycles, harmonics=harmonics)
    weights = np.sqrt(_taper_weights(len(profile)))
    coefficients, _, _, _ = np.linalg.lstsq(
        design * weights[:, None], profile * weights, rcond=None
    )
    fitted_profile = design @ coefficients
    residuals = profile - fitted_profile

    return SquareWaveFit(
        terms=len(harmonics),
        cycles=cycles,
        harmonics=harmonics,
        offset=float(coefficients[0]),
        baseline_slope=float(coefficients[1]),
        sine_coefficients=np.asarray(coefficients[2::2], dtype=np.float64),
        cosine_coefficients=np.asarray(coefficients[3::2], dtype=np.float64),
        fitted_profile=fitted_profile,
        residuals=residuals,
        residual_rms=float(np.sqrt(np.sum(weights**2 * residuals**2) / np.sum(weights**2))),
    )


def _taper_weights(sample_count: int) -> np.ndarray:
    """Tukey window: flat in the middle, cosine roll-off over TAPER_FRACTION of the samples."""
    x = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    edge = TAPER_FRACTION / 2.0
    weights = np.ones(sample_count)
    if edge > 0:
        rising, falling = x < edge, x > 1.0 - edge
        weights[rising] = 0.5 * (1.0 - np.cos(np.pi * x[rising] / edge))
        weights[falling] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - x[falling]) / edge))
    return weights


def _square_wave_design_matrix(
    sample_count: int,
    *,
    cycles: float,
    harmonics: np.ndarray | None = None,
) -> np.ndarray:
    if harmonics is None:
        harmonics = _odd_harmonics()
    _validate_sample_count(sample_count, parameter_count=2 + 2 * len(harmonics))

    x = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    centered_x = x - 0.5
    columns = [np.ones(sample_count, dtype=np.float64), centered_x]
    for harmonic in harmonics:
        radians = 2.0 * np.pi * cycles * float(harmonic) * x
        columns.append(np.sin(radians))
        columns.append(np.cos(radians))
    return np.column_stack(columns)


def _odd_harmonics() -> np.ndarray:
    return np.arange(1, 2 * SQUARE_WAVE_TERMS, 2, dtype=np.int64)


def _as_1d_finite_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError(f"profile is {profile.ndim}D; expected a 1D array")
    if not np.all(np.isfinite(profile)):
        raise ValueError("profile must contain only finite values")
    return profile


def _validate_sample_count(sample_count: int, *, parameter_count: int) -> None:
    if sample_count < parameter_count:
        raise ValueError(
            f"profile needs at least {parameter_count} samples for this square-wave fit, "
            f"got {sample_count}"
        )
