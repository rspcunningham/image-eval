from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from image_eval.mtf_profiles import BarROIProfile


SQUARE_WAVE_TERMS = 3
BAR_PROFILE_CYCLES = 3.0
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
    *,
    terms: int = SQUARE_WAVE_TERMS,
    cycles: float | None = None,
    cycle_search_min: float = CYCLE_SEARCH_MIN,
    cycle_search_max: float = CYCLE_SEARCH_MAX,
    cycle_search_steps: int = CYCLE_SEARCH_STEPS,
) -> SquareWaveFit:
    harmonics = odd_harmonics(terms)
    profile = _as_1d_finite_profile(profile, parameter_count=2 + 2 * len(harmonics))
    if cycles is None:
        return _best_cycle_fit(
            profile,
            harmonics=harmonics,
            terms=terms,
            cycle_search_min=cycle_search_min,
            cycle_search_max=cycle_search_max,
            cycle_search_steps=cycle_search_steps,
        )

    return _fit_square_wave_profile_at_cycles(
        profile,
        harmonics=harmonics,
        terms=terms,
        cycles=cycles,
    )


def _best_cycle_fit(
    profile: np.ndarray,
    *,
    harmonics: np.ndarray,
    terms: int,
    cycle_search_min: float,
    cycle_search_max: float,
    cycle_search_steps: int,
) -> SquareWaveFit:
    if not np.isfinite(cycle_search_min) or not np.isfinite(cycle_search_max):
        raise ValueError("cycle search bounds must be finite")
    if cycle_search_min <= 0 or cycle_search_max <= 0 or cycle_search_max < cycle_search_min:
        raise ValueError(
            "cycle search bounds must be positive with cycle_search_max >= cycle_search_min"
        )
    if not isinstance(cycle_search_steps, int) or isinstance(cycle_search_steps, bool):
        raise ValueError(f"cycle_search_steps must be an integer, got {cycle_search_steps}")
    if cycle_search_steps < 1:
        raise ValueError(f"cycle_search_steps must be positive, got {cycle_search_steps}")

    best_fit: SquareWaveFit | None = None
    for candidate_cycles in np.linspace(cycle_search_min, cycle_search_max, cycle_search_steps):
        fit = _fit_square_wave_profile_at_cycles(
            profile,
            harmonics=harmonics,
            terms=terms,
            cycles=float(candidate_cycles),
        )
        if best_fit is None or fit.residual_rms < best_fit.residual_rms:
            best_fit = fit

    if best_fit is None:
        raise RuntimeError("could not fit square-wave profile")
    return best_fit


def _fit_square_wave_profile_at_cycles(
    profile: np.ndarray,
    *,
    harmonics: np.ndarray,
    terms: int,
    cycles: float,
) -> SquareWaveFit:
    design = square_wave_design_matrix(len(profile), terms=terms, cycles=cycles)
    coefficients, _, _, _ = np.linalg.lstsq(design, profile, rcond=None)
    fitted_profile = design @ coefficients
    residuals = profile - fitted_profile

    return SquareWaveFit(
        terms=terms,
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


def square_wave_design_matrix(
    sample_count: int,
    *,
    terms: int = SQUARE_WAVE_TERMS,
    cycles: float = BAR_PROFILE_CYCLES,
) -> np.ndarray:
    harmonics = odd_harmonics(terms)
    _validate_sample_count(sample_count, parameter_count=2 + 2 * len(harmonics))
    if not np.isfinite(cycles) or cycles <= 0:
        raise ValueError(f"cycles must be positive and finite, got {cycles}")

    x = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    centered_x = x - 0.5
    columns = [np.ones(sample_count, dtype=np.float64), centered_x]
    for harmonic in harmonics:
        radians = 2.0 * np.pi * cycles * float(harmonic) * x
        columns.append(np.sin(radians))
        columns.append(np.cos(radians))
    return np.column_stack(columns)


def odd_harmonics(terms: int = SQUARE_WAVE_TERMS) -> np.ndarray:
    if not isinstance(terms, int) or isinstance(terms, bool) or terms < 1:
        raise ValueError(f"terms must be a positive integer, got {terms}")
    return np.arange(1, 2 * terms, 2, dtype=np.int64)


def _as_1d_finite_profile(profile: np.ndarray, *, parameter_count: int) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError(f"profile is {profile.ndim}D; expected a 1D array")
    if not np.all(np.isfinite(profile)):
        raise ValueError("profile must contain only finite values")
    _validate_sample_count(len(profile), parameter_count=parameter_count)
    return profile


def _validate_sample_count(sample_count: int, *, parameter_count: int) -> None:
    if sample_count < parameter_count:
        raise ValueError(
            f"profile needs at least {parameter_count} samples for this square-wave fit, "
            f"got {sample_count}"
        )
