from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from image_eval.mtf_profiles import BarROIProfile, OrientedSamples


SQUARE_WAVE_TERMS = 3
CYCLE_SEARCH_MIN = 2.0
CYCLE_SEARCH_MAX = 4.0
# Coarse grid over the search range, then a fine grid one coarse step either side of the best.
CYCLE_COARSE_STEP = 0.05
CYCLE_FINE_STEP = 0.005
# Fraction of each element's width (split across both ends) down-weighted by a Tukey
# taper, so edge halos beyond the outer bars carry little weight in the fit.
TAPER_FRACTION = 0.5
# Free-period fits used to calibrate pixels per millimetre must at least reach this
# fundamental MTF; poorly resolved elements would otherwise fit noise at any period.
CALIBRATION_MIN_MTF = 0.3


@dataclass(frozen=True)
class SquareWaveFit:
    terms: int
    cycles: float
    harmonics: np.ndarray
    offset: float
    baseline_slope: float
    sine_coefficients: np.ndarray
    cosine_coefficients: np.ndarray
    fitted_values: np.ndarray
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
    elements, then each ROI is refit with its cycle count fixed, so poorly
    resolved elements cannot pick whichever period best matches their noise. The fixed
    fits include every odd harmonic the sampling can carry; which of those are resolved
    is decided against the flat-region noise floor in mtf_results.
    """
    free_fits = [
        fit_square_wave(
            roi.samples,
            harmonics=harmonics_below_nyquist(CYCLE_SEARCH_MAX, roi.samples.width_px),
        )
        for roi in roi_profiles
    ]
    pixels_per_mm = calibrate_pixels_per_mm(roi_profiles, free_fits)
    return fit_bar_roi_profiles_at_scale(roi_profiles, pixels_per_mm)


def fit_bar_roi_profiles_at_scale(
    roi_profiles: Sequence[BarROIProfile], pixels_per_mm: float
) -> list[FittedBarROIProfile]:
    return [
        FittedBarROIProfile(
            roi_profile=roi,
            fit=fit_square_wave_at_cycles(
                roi.samples,
                cycles=cycles,
                harmonics=harmonics_below_nyquist(cycles, roi.samples.width_px),
            ),
        )
        for roi in roi_profiles
        for cycles in [roi.frequency_lp_per_mm * roi.samples.width_px / pixels_per_mm]
    ]


def harmonics_below_nyquist(cycles: float, width_px: float) -> np.ndarray:
    """Odd harmonics to try: the fundamental, plus each harmonic below the sampling Nyquist.

    An element width_px pixels wide spanning cycles periods carries harmonic h only if
    h * cycles < width_px / 2; higher harmonics alias and would only fit noise.
    """
    harmonics = _odd_harmonics()
    keep = (harmonics == 1) | (harmonics * cycles < width_px / 2.0)
    return harmonics[keep]


def calibrate_pixels_per_mm(
    roi_profiles: Sequence[BarROIProfile],
    free_fits: Sequence[SquareWaveFit],
) -> float:
    if not roi_profiles:
        raise ValueError("cannot calibrate pixels per millimetre without ROI profiles")
    candidates = [
        roi_profile.frequency_lp_per_mm * roi_profile.samples.width_px / fit.cycles
        for roi_profile, fit in zip(roi_profiles, free_fits, strict=True)
        if CYCLE_SEARCH_MIN < fit.cycles < CYCLE_SEARCH_MAX
        and fit.fundamental_amplitude * np.pi / 2.0 >= CALIBRATION_MIN_MTF
    ]
    if not candidates:
        candidates = [
            roi_profile.frequency_lp_per_mm * roi_profile.samples.width_px / fit.cycles
            for roi_profile, fit in zip(roi_profiles, free_fits, strict=True)
        ]
    return float(np.median(candidates))


def fit_square_wave(
    samples: OrientedSamples, *, harmonics: np.ndarray | None = None
) -> SquareWaveFit:
    """Best fit over a coarse-then-fine grid of cycle counts."""
    harmonics = _odd_harmonics() if harmonics is None else harmonics

    def best(grid: np.ndarray) -> SquareWaveFit:
        fits = (
            fit_square_wave_at_cycles(samples, cycles=float(cycles), harmonics=harmonics)
            for cycles in grid
        )
        return min(fits, key=lambda fit: fit.residual_rms)

    coarse = best(np.arange(CYCLE_SEARCH_MIN, CYCLE_SEARCH_MAX + 1e-9, CYCLE_COARSE_STEP))
    return best(
        np.clip(
            coarse.cycles
            + np.arange(-CYCLE_COARSE_STEP, CYCLE_COARSE_STEP + 1e-9, CYCLE_FINE_STEP),
            CYCLE_SEARCH_MIN,
            CYCLE_SEARCH_MAX,
        )
    )


def fit_square_wave_at_cycles(
    samples: OrientedSamples, *, cycles: float, harmonics: np.ndarray | None = None
) -> SquareWaveFit:
    """Weighted least squares of offset, linear baseline and odd harmonics at known period.

    positions are fractions of the element width, so a pixel anywhere along a rotated bar
    is fit at its true distance across the bars.
    """
    harmonics = _odd_harmonics() if harmonics is None else harmonics
    x, values = samples.positions, samples.values
    if len(x) < 2 + 2 * len(harmonics):
        raise ValueError(f"square-wave fit needs more samples than the {len(x)} given")
    columns = [np.ones_like(x), x - 0.5]
    for harmonic in harmonics:
        radians = 2.0 * np.pi * cycles * float(harmonic) * x
        columns += [np.sin(radians), np.cos(radians)]
    design = np.column_stack(columns)
    weights = _taper_weights(x)
    root_weights = np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(
        design * root_weights[:, None], values * root_weights, rcond=None
    )
    fitted_values = design @ coefficients
    residuals = values - fitted_values

    return SquareWaveFit(
        terms=len(harmonics),
        cycles=cycles,
        harmonics=harmonics,
        offset=float(coefficients[0]),
        baseline_slope=float(coefficients[1]),
        sine_coefficients=np.asarray(coefficients[2::2], dtype=np.float64),
        cosine_coefficients=np.asarray(coefficients[3::2], dtype=np.float64),
        fitted_values=fitted_values,
        residuals=residuals,
        residual_rms=float(np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))),
    )


def _taper_weights(x: np.ndarray) -> np.ndarray:
    """Tukey window: flat in the middle, cosine roll-off over TAPER_FRACTION of the width."""
    edge = TAPER_FRACTION / 2.0
    distance = np.clip(np.minimum(x, 1.0 - x), 0.0, None)
    return np.where(distance < edge, 0.5 * (1.0 - np.cos(np.pi * distance / edge)), 1.0)


def _odd_harmonics() -> np.ndarray:
    return np.arange(1, 2 * SQUARE_WAVE_TERMS, 2, dtype=np.int64)
