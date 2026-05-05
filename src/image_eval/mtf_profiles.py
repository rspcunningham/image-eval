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
class NormalizedImage:
    image: np.ndarray
    normalization: IntensityNormalization


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
    normalized = normalize_image_intensity(image, template.get("normalization_rois"))
    return PreparedMTFProfiles(
        normalized_image=normalized.image,
        normalization=normalized.normalization,
        bar_profiles=bar_roi_profiles(normalized.image, template.get("bar_rois")),
    )


def normalize_image_intensity(image: np.ndarray, normalization_rois: Any) -> NormalizedImage:
    if not isinstance(normalization_rois, dict):
        raise ValueError("template does not contain a normalization_rois object")
    normalization_rois = cast(dict[str, Any], normalization_rois)

    image = as_2d_float_image(image)
    black_rect = as_rect(normalization_rois.get("black"), "normalization_rois.black")
    white_rect = as_rect(normalization_rois.get("white"), "normalization_rois.white")

    black_mean = float(np.mean(finite_roi_pixels(image, black_rect, "normalization_rois.black")))
    white_mean = float(np.mean(finite_roi_pixels(image, white_rect, "normalization_rois.white")))
    scale = white_mean - black_mean
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("white normalization ROI mean must be greater than black ROI mean")

    return NormalizedImage(
        image=(image - black_mean) / scale,
        normalization=IntensityNormalization(
            black_mean=black_mean,
            white_mean=white_mean,
        ),
    )


def bar_roi_profiles(normalized_image: np.ndarray, bar_rois: Any) -> list[BarROIProfile]:
    if not isinstance(bar_rois, list):
        raise ValueError("template does not contain a bar_rois array")

    image = as_2d_float_image(normalized_image)
    profiles: list[BarROIProfile] = []
    for index, roi in enumerate(bar_rois):
        if not isinstance(roi, dict):
            raise ValueError(f"bar_rois[{index}] must be a JSON object")
        roi = cast(dict[str, Any], roi)

        rect_value = roi.get("rect")
        if rect_value is None:
            continue

        group = as_int(roi.get("group"), f"bar_rois[{index}].group")
        element = as_int(roi.get("element"), f"bar_rois[{index}].element")
        orientation = roi.get("orientation")
        if not isinstance(orientation, str):
            raise ValueError(f"bar_rois[{index}].orientation must be a string")

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

