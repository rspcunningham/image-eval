from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np

from image_eval.mtf_profiles import prepare_mtf_profiles
from image_eval.square_wave_fit import FittedBarROIProfile, fit_bar_roi_profiles


FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION = math.pi / 2.0


@dataclass(frozen=True)
class MTFResult:
    frequency_lp_per_mm: float
    x_mtf: float | None
    y_mtf: float | None
    average_mtf: float


class MTFReport(NamedTuple):
    results: list[MTFResult]
    fitted_profiles: list[FittedBarROIProfile]


def calculate_mtf_results(image: np.ndarray, template: dict[str, Any]) -> list[MTFResult]:
    return calculate_mtf_report(image, template).results


def calculate_mtf_report(image: np.ndarray, template: dict[str, Any]) -> MTFReport:
    prepared = prepare_mtf_profiles(image, template)
    fitted_profiles = fit_bar_roi_profiles(prepared.bar_profiles)
    return MTFReport(
        results=mtf_results_from_fits(fitted_profiles),
        fitted_profiles=fitted_profiles,
    )


def mtf_results_from_fits(fitted_profiles: Sequence[FittedBarROIProfile]) -> list[MTFResult]:
    grouped: dict[float, dict[str, list[float]]] = {}

    for fitted_profile in fitted_profiles:
        roi = fitted_profile.roi_profile
        orientation = roi.orientation
        if orientation not in ("X", "Y"):
            raise ValueError(f"bar ROI orientation must be X or Y, got {orientation}")

        orientation_values = grouped.setdefault(
            roi.frequency_lp_per_mm,
            {"X": [], "Y": []},
        )
        orientation_values[orientation].append(roi_mtf_value(fitted_profile))

    results: list[MTFResult] = []
    for frequency, orientation_values in sorted(grouped.items()):
        x_mtf = _mean_or_none(orientation_values["X"])
        y_mtf = _mean_or_none(orientation_values["Y"])
        average_mtf = _mean_available(x_mtf, y_mtf)
        results.append(
            MTFResult(
                frequency_lp_per_mm=frequency,
                x_mtf=x_mtf,
                y_mtf=y_mtf,
                average_mtf=average_mtf,
            )
        )

    return results


def roi_mtf_value(fitted_profile: FittedBarROIProfile) -> float:
    return fitted_profile.fit.fundamental_amplitude * FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION


def roi_pixels_per_mm(fitted_profile: FittedBarROIProfile) -> float:
    roi = fitted_profile.roi_profile
    profile_pixels = len(roi.profile)
    fitted_cycles = fitted_profile.fit.cycles
    known_lp_per_mm = roi.frequency_lp_per_mm

    if profile_pixels <= 0:
        raise ValueError("ROI profile must contain at least one pixel")
    if not np.isfinite(fitted_cycles) or fitted_cycles <= 0:
        raise ValueError("fitted ROI cycles must be positive and finite")
    if not np.isfinite(known_lp_per_mm) or known_lp_per_mm <= 0:
        raise ValueError("ROI frequency_lp_per_mm must be positive and finite")

    return float(known_lp_per_mm * profile_pixels / fitted_cycles)


def average_pixels_per_mm_from_fits(fitted_profiles: Sequence[FittedBarROIProfile]) -> float:
    values = [roi_pixels_per_mm(fitted_profile) for fitted_profile in fitted_profiles]
    if not values:
        raise ValueError("cannot calculate pixels per millimetre without fitted ROI profiles")
    return float(np.mean(values))


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _mean_available(*values: float | None) -> float:
    available = [value for value in values if value is not None]
    if not available:
        raise ValueError("cannot calculate average MTF without X or Y values")
    return float(np.mean(available))

