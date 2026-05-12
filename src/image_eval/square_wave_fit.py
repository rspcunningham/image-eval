from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from image_eval.mtf_profiles import BarROIProfile


SQUARE_WAVE_TERMS = 3
CYCLE_SEARCH_MIN = 2.0
CYCLE_SEARCH_MAX = 4.0
CYCLE_SEARCH_STEPS = 401


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
    return [fit_bar_roi_profile(roi_profile) for roi_profile in roi_profiles]


def fit_bar_roi_profile(roi_profile: BarROIProfile) -> FittedBarROIProfile:
    return FittedBarROIProfile(
        roi_profile=roi_profile,
        fit=fit_square_wave_profile(roi_profile.profile),
    )


def fit_square_wave_profile(
    profile: np.ndarray,
) -> SquareWaveFit:
    harmonics = _odd_harmonics()
    profile = _as_1d_finite_profile(profile)
    return _best_cycle_fit(
        profile,
        harmonics=harmonics,
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
    design = _square_wave_design_matrix(len(profile), cycles=cycles)
    coefficients, _, _, _ = np.linalg.lstsq(design, profile, rcond=None)
    fitted_profile = design @ coefficients
    residuals = profile - fitted_profile

    return SquareWaveFit(
        terms=SQUARE_WAVE_TERMS,
        cycles=cycles,
        harmonics=harmonics,
        offset=float(coefficients[0]),
        baseline_slope=float(coefficients[1]),
        sine_coefficients=np.asarray(coefficients[2::2], dtype=np.float64),
        cosine_coefficients=np.asarray(coefficients[3::2], dtype=np.float64),
        fitted_profile=fitted_profile,
        residuals=residuals,
        residual_rms=float(np.sqrt(np.mean(residuals * residuals))),
    )


def _square_wave_design_matrix(
    sample_count: int,
    *,
    cycles: float,
) -> np.ndarray:
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
    _validate_sample_count(len(profile), parameter_count=2 + 2 * SQUARE_WAVE_TERMS)
    return profile


def _validate_sample_count(sample_count: int, *, parameter_count: int) -> None:
    if sample_count < parameter_count:
        raise ValueError(
            f"profile needs at least {parameter_count} samples for this square-wave fit, "
            f"got {sample_count}"
        )
