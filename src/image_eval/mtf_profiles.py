from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from image_eval.roi import (
    Rect,
    as_2d_float_image,
    as_int,
    as_rect,
    crop_image,
    finite_roi_pixels,
)
from image_eval.usaf_1951 import line_pairs_per_mm


ProfileAxis = Literal["x", "y"]


@dataclass(frozen=True)
class IntensityNormalization:
    black_mean: float
    white_mean: float


@dataclass(frozen=True)
class NormalizationROIs:
    black: Rect
    white: Rect


@dataclass(frozen=True)
class NormalizedImage:
    image: np.ndarray
    normalization: IntensityNormalization
    normalization_rois: NormalizationROIs


@dataclass(frozen=True)
class BarROIProfile:
    group: int
    element: int
    orientation: str
    frequency_lp_per_mm: float
    rect: Rect
    profile_axis: ProfileAxis
    profile: np.ndarray


@dataclass(frozen=True)
class PreparedMTFProfiles:
    normalized_image: np.ndarray
    normalization: IntensityNormalization
    bar_profiles: list[BarROIProfile]


def prepare_mtf_profiles(image: np.ndarray, template: dict[str, Any]) -> PreparedMTFProfiles:
    normalized = normalize_image_intensity(image, template["normalization_rois"])
    return PreparedMTFProfiles(
        normalized_image=normalized.image,
        normalization=normalized.normalization,
        bar_profiles=bar_roi_profiles(normalized.image, template["bar_rois"]),
    )


def normalize_image_intensity(image: np.ndarray, normalization_rois: Any) -> NormalizedImage:
    image = as_2d_float_image(image)
    rois = normalization_roi_rects(normalization_rois)

    black_mean = normalization_roi_mean(image, rois.black, "black")
    white_mean = normalization_roi_mean(image, rois.white, "white")
    scale = white_mean - black_mean
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("white normalization ROI mean must differ from black ROI mean")

    return NormalizedImage(
        image=(image - black_mean) / scale,
        normalization=IntensityNormalization(
            black_mean=black_mean,
            white_mean=white_mean,
        ),
        normalization_rois=rois,
    )


def normalization_roi_rects(normalization_rois: Any) -> NormalizationROIs:
    normalization_rois = cast(dict[str, Any], normalization_rois)
    return NormalizationROIs(
        black=as_rect(normalization_rois["black"], "normalization_rois.black"),
        white=as_rect(normalization_rois["white"], "normalization_rois.white"),
    )


def normalization_roi_mean(image: np.ndarray, rect: Rect, name: str) -> float:
    return float(np.mean(finite_roi_pixels(image, rect, f"normalization_rois.{name}")))


def bar_roi_profiles(normalized_image: np.ndarray, bar_rois: Any) -> list[BarROIProfile]:
    image = as_2d_float_image(normalized_image)
    profiles: list[BarROIProfile] = []
    for index, roi in enumerate(bar_rois):
        rect_value = roi["rect"]
        if rect_value is None:
            continue

        group = as_int(roi["group"], f"bar_rois[{index}].group")
        element = as_int(roi["element"], f"bar_rois[{index}].element")
        orientation = cast(str, roi["orientation"])
        rect = as_rect(rect_value, f"bar_rois[{index}].rect")
        crop = crop_image(image, rect, f"bar_rois[{index}].rect")
        profile_axis, collapse_axis = _profile_axes(orientation)

        profiles.append(
            BarROIProfile(
                group=group,
                element=element,
                orientation=orientation,
                frequency_lp_per_mm=line_pairs_per_mm(group, element),
                rect=rect,
                profile_axis=profile_axis,
                profile=np.mean(crop, axis=collapse_axis),
            )
        )

    return profiles


def _profile_axes(orientation: str) -> tuple[ProfileAxis, int]:
    if orientation == "X":
        return "x", 0
    if orientation == "Y":
        return "y", 1
    raise ValueError(f"bar ROI orientation must be X or Y, got {orientation}")
