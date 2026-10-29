from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np

from image_eval.mtf_profiles import (
    NormalizationSpec,
    normalization_roi_rects,
    prepare_mtf_profiles,
)
from image_eval.roi import Rect, crop_image
from image_eval.square_wave_fit import (
    FittedBarROIProfile,
    SquareWaveFit,
    fit_bar_roi_profiles,
    fit_square_wave_profile,
    fit_square_wave_profile_at_cycles,
)

# An element counts as resolved only if its MTF exceeds the MTF measured on flat
# (bar-free) regions at the same period by this factor, and also exceeds an absolute
# contrast floor: smooth reconstructions can have near-zero flat-region noise while
# still carrying structured artifacts among the bars.
NOISE_FLOOR_FACTOR = 3.0
MIN_RESOLVED_MTF = 0.1


@dataclass(frozen=True)
class MTFResult:
    cycles_per_mm: float
    orientation: str
    mtf: float
    noise_mtf: float | None = None
    resolved: bool | None = None


class MTFReport(NamedTuple):
    results: list[MTFResult]
    fitted_profiles: list[FittedBarROIProfile]
    pixels_per_mm: float


def calculate_mtf_results(
    image: np.ndarray,
    template: dict[str, Any],
    *,
    norm: NormalizationSpec = "mean",
) -> list[MTFResult]:
    return calculate_mtf_report(image, template, norm=norm).results


def calculate_mtf_report(
    image: np.ndarray,
    template: dict[str, Any],
    *,
    norm: NormalizationSpec = "mean",
) -> MTFReport:
    prepared = prepare_mtf_profiles(image, template, norm=norm)
    fitted_profiles = fit_bar_roi_profiles(prepared.bar_profiles)
    pixels_per_mm = average_pixels_per_mm_from_fits(fitted_profiles)
    flat_rects = normalization_roi_rects(template["normalization_rois"])

    def noise_floors(profiles: Sequence[FittedBarROIProfile]) -> list[dict[float, float] | None]:
        return [
            flat_region_mtf_points(
                prepared.normalized_image,
                [flat_rects.black, flat_rects.white],
                fitted_profile,
                pixels_per_mm,
            )
            for fitted_profile in profiles
        ]

    # Square waves are built from the odd harmonics that are actually resolved: fit every
    # harmonic below Nyquist, then refit without the 3rd and 5th wherever they are not.
    fitted_profiles = [
        drop_unresolved_harmonics(fitted_profile, noise)
        for fitted_profile, noise in zip(
            fitted_profiles, noise_floors(fitted_profiles), strict=True
        )
    ]
    noise_points = noise_floors(fitted_profiles)
    return MTFReport(
        results=mtf_results_from_fits(fitted_profiles, noise_points),
        fitted_profiles=fitted_profiles,
        pixels_per_mm=pixels_per_mm,
    )


def mtf_results_from_fits(
    fitted_profiles: Sequence[FittedBarROIProfile],
    noise_points: Sequence[dict[float, float] | None] | None = None,
) -> list[MTFResult]:
    if noise_points is None:
        noise_points = [None] * len(fitted_profiles)
    return [
        result
        for fitted_profile, noise in zip(fitted_profiles, noise_points, strict=True)
        for result in fitted_profile_mtf_results(fitted_profile, noise)
    ]


def flat_region_mtf_points(
    normalized_image: np.ndarray,
    flat_rects: Sequence[Rect],
    fitted_profile: FittedBarROIProfile,
    pixels_per_mm: float,
) -> dict[float, float] | None:
    """MTF that the same fixed-period fit reports on bar-free regions: the noise floor.

    Each flat region is cropped to at most the bar ROI's size, collapsed along the same
    axis, and fit at the bar ROI's period. The larger value across regions is kept.
    """
    roi = fitted_profile.roi_profile
    roi_height = roi.rect["y1"] - roi.rect["y0"]
    roi_width = roi.rect["x1"] - roi.rect["x0"]
    floors: dict[float, float] | None = None
    for rect in flat_rects:
        crop = crop_image(normalized_image, rect, "normalization ROI")
        crop = crop[: min(roi_height, crop.shape[0]), : min(roi_width, crop.shape[1])]
        profile = np.mean(crop, axis=0 if roi.profile_axis == "x" else 1)
        try:
            fit = fit_square_wave_profile_at_cycles(
                profile,
                cycles=roi.frequency_lp_per_mm * len(profile) / pixels_per_mm,
                harmonics=fitted_profile.fit.harmonics,
            )
        except ValueError:  # flat region too short along the profile axis
            continue
        points = fitted_square_wave_mtf_points(fit, roi.frequency_lp_per_mm)
        floors = points if floors is None else {f: max(floors[f], v) for f, v in points.items()}
    return floors


def line_profile_mtf_points(
    profile: np.ndarray,
    fundamental_cycles_per_mm: float,
) -> dict[float, float]:
    return fitted_square_wave_mtf_points(
        fit_square_wave_profile(profile),
        fundamental_cycles_per_mm,
    )


def is_resolved(mtf: float, noise_mtf: float) -> bool:
    return bool(mtf > max(NOISE_FLOOR_FACTOR * noise_mtf, MIN_RESOLVED_MTF))


def drop_unresolved_harmonics(
    fitted_profile: FittedBarROIProfile,
    noise_points: dict[float, float] | None,
) -> FittedBarROIProfile:
    """Refit without the 3rd and 5th harmonics that do not clear the noise floor.

    The fundamental is always kept. Without a measurable noise floor no harmonic is
    trusted, so only the fundamental remains.
    """
    roi = fitted_profile.roi_profile
    fit = fitted_profile.fit
    mtf_points = fitted_square_wave_mtf_points(fit, roi.frequency_lp_per_mm)
    kept = np.array(
        [
            harmonic
            for harmonic, (cycles_per_mm, mtf) in zip(
                fit.harmonics, mtf_points.items(), strict=True
            )
            if harmonic == 1
            or (noise_points is not None and is_resolved(mtf, noise_points[cycles_per_mm]))
        ],
        dtype=np.int64,
    )
    if len(kept) == len(fit.harmonics):
        return fitted_profile
    return FittedBarROIProfile(
        roi_profile=roi,
        fit=fit_square_wave_profile_at_cycles(roi.profile, cycles=fit.cycles, harmonics=kept),
    )


def fitted_profile_mtf_results(
    fitted_profile: FittedBarROIProfile,
    noise_points: dict[float, float] | None = None,
) -> list[MTFResult]:
    roi = fitted_profile.roi_profile
    return [
        MTFResult(
            cycles_per_mm=cycles_per_mm,
            orientation=roi.orientation,
            mtf=mtf,
            noise_mtf=None if noise_points is None else noise_points[cycles_per_mm],
            resolved=None
            if noise_points is None
            else is_resolved(mtf, noise_points[cycles_per_mm]),
        )
        for cycles_per_mm, mtf in fitted_square_wave_mtf_points(
            fitted_profile.fit,
            roi.frequency_lp_per_mm,
        ).items()
    ]


def fitted_square_wave_mtf_points(
    fit: SquareWaveFit,
    fundamental_cycles_per_mm: float,
) -> dict[float, float]:
    return {
        float(fundamental_cycles_per_mm * harmonic): float(
            np.hypot(sine, cosine) * math.pi * harmonic / 2.0
        )
        for harmonic, sine, cosine in zip(
            fit.harmonics,
            fit.sine_coefficients,
            fit.cosine_coefficients,
            strict=True,
        )
    }


def roi_pixels_per_mm(fitted_profile: FittedBarROIProfile) -> float:
    roi = fitted_profile.roi_profile
    return float(roi.frequency_lp_per_mm * len(roi.profile) / fitted_profile.fit.cycles)


def average_pixels_per_mm_from_fits(fitted_profiles: Sequence[FittedBarROIProfile]) -> float:
    values = [roi_pixels_per_mm(fitted_profile) for fitted_profile in fitted_profiles]
    if not values:
        raise ValueError("cannot calculate pixels per millimetre without fitted ROI profiles")
    return float(np.mean(values))
