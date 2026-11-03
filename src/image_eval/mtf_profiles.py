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


NormalizationSpec = Literal["mean"] | float
# Middle fraction of the bar length kept: averaging over the blurred bar ends would dilute
# contrast equally at every harmonic, most in small elements.
BAR_LENGTH_FRACTION = 0.6


@dataclass(frozen=True)
class IntensityNormalization:
    black_level: float
    white_level: float
    method: str
    cutoff: float | None = None

    @property
    def black_mean(self) -> float:
        return self.black_level

    @property
    def white_mean(self) -> float:
        return self.white_level


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
class OrientedSamples:
    """Pixels of a rotated element, positioned across the bars as a fraction of its width."""

    positions: np.ndarray
    values: np.ndarray
    width_px: float


@dataclass(frozen=True)
class BarROIProfile:
    group: int
    element: int
    orientation: str
    frequency_lp_per_mm: float
    rect: Rect
    samples: OrientedSamples


@dataclass(frozen=True)
class PreparedMTFProfiles:
    normalized_image: np.ndarray
    normalization: IntensityNormalization
    bar_profiles: list[BarROIProfile]


def prepare_mtf_profiles(
    image: np.ndarray,
    template: dict[str, Any],
    *,
    norm: NormalizationSpec = "mean",
    angle_degrees: float = 0.0,
) -> PreparedMTFProfiles:
    normalized = normalize_image_intensity(image, template["normalization_rois"], norm=norm)
    return PreparedMTFProfiles(
        normalized_image=normalized.image,
        normalization=normalized.normalization,
        bar_profiles=bar_roi_profiles(normalized.image, template["bar_rois"], angle_degrees),
    )


def normalize_image_intensity(
    image: np.ndarray,
    normalization_rois: Any,
    *,
    norm: NormalizationSpec = "mean",
) -> NormalizedImage:
    image = as_2d_float_image(image)
    rois = normalization_roi_rects(normalization_rois)

    black_level, white_level, normalization = normalization_levels(image, rois, norm=norm)
    scale = white_level - black_level
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("white normalization level must differ from black normalization level")

    return NormalizedImage(
        image=(image - black_level) / scale,
        normalization=normalization,
        normalization_rois=rois,
    )


def normalization_levels(
    image: np.ndarray,
    rois: NormalizationROIs,
    *,
    norm: NormalizationSpec = "mean",
) -> tuple[float, float, IntensityNormalization]:
    black_pixels = finite_roi_pixels(image, rois.black, "normalization_rois.black")
    white_pixels = finite_roi_pixels(image, rois.white, "normalization_rois.white")

    if norm == "mean":
        black_level = float(np.mean(black_pixels))
        white_level = float(np.mean(white_pixels))
        return (
            black_level,
            white_level,
            IntensityNormalization(
                black_level=black_level,
                white_level=white_level,
                method="mean",
            ),
        )

    cutoff = normalization_cutoff(norm)
    black_level = float(np.quantile(black_pixels, cutoff))
    white_level = float(np.quantile(white_pixels, 1.0 - cutoff))
    return (
        black_level,
        white_level,
        IntensityNormalization(
            black_level=black_level,
            white_level=white_level,
            method="quantile",
            cutoff=cutoff,
        ),
    )


def normalization_cutoff(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("normalization must be 'mean' or a numeric cutoff")

    cutoff = float(value)
    if not np.isfinite(cutoff) or cutoff < 0 or cutoff >= 0.5:
        raise ValueError("numeric normalization cutoff must be >= 0 and < 0.5")
    return cutoff


def normalization_roi_rects(normalization_rois: Any) -> NormalizationROIs:
    if not isinstance(normalization_rois, dict):
        raise ValueError("normalization_rois must be a JSON object")
    normalization_rois = cast(dict[str, Any], normalization_rois)
    return NormalizationROIs(
        black=as_rect(normalization_rois["black"], "normalization_rois.black"),
        white=as_rect(normalization_rois["white"], "normalization_rois.white"),
    )


def normalization_roi_mean(image: np.ndarray, rect: Rect, name: str) -> float:
    return float(np.mean(finite_roi_pixels(image, rect, f"normalization_rois.{name}")))


def bar_roi_profiles(
    normalized_image: np.ndarray, bar_rois: Any, angle_degrees: float
) -> list[BarROIProfile]:
    image = as_2d_float_image(normalized_image)
    profiles: list[BarROIProfile] = []
    for index, roi in enumerate(bar_rois):
        rect_value = roi["rect"]
        if rect_value is None:
            continue

        orientation = cast(str, roi["orientation"])
        rect = as_rect(rect_value, f"bar_rois[{index}].rect")
        group = as_int(roi["group"], f"bar_rois[{index}].group")
        element = as_int(roi["element"], f"bar_rois[{index}].element")
        profiles.append(
            BarROIProfile(
                group=group,
                element=element,
                orientation=orientation,
                frequency_lp_per_mm=line_pairs_per_mm(group, element),
                rect=rect,
                samples=oriented_samples(image, rect, orientation, angle_degrees),
            )
        )

    return profiles


def oriented_samples(
    image: np.ndarray, rect: Rect, orientation: str, angle_degrees: float
) -> OrientedSamples:
    """Pixels of the element inside rect, with bars rotated by angle_degrees.

    rect is the axis-aligned bounding box of the rotated element. Each pixel is placed by
    its exact distance across the bars, so nothing is resampled; along the bars only the
    middle BAR_LENGTH_FRACTION is kept. Positive angles turn the bars clockwise on screen.
    """
    if orientation not in ("X", "Y"):
        raise ValueError(f"bar ROI orientation must be X or Y, got {orientation}")
    crop = crop_image(image, rect, "bar ROI")
    angle = np.radians(angle_degrees)
    cos, sin = np.cos(angle), abs(np.sin(angle))
    box_width, box_height = rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]
    box_across, box_along = (
        (box_width, box_height) if orientation == "X" else (box_height, box_width)
    )
    # Invert the bounding box of a rotated across x along rectangle.
    across = (box_across * cos - box_along * sin) / (cos**2 - sin**2)
    along = (box_along * cos - box_across * sin) / (cos**2 - sin**2)

    dy, dx = np.mgrid[: crop.shape[0], : crop.shape[1]] + 0.5
    dx, dy = dx - box_width / 2, dy - box_height / 2
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    u_across, v_along = (u, v) if orientation == "X" else (v, u)
    keep = (np.abs(u_across) < across / 2) & (np.abs(v_along) < BAR_LENGTH_FRACTION * along / 2)
    order = np.argsort(u_across[keep])
    return OrientedSamples(
        positions=(u_across[keep][order] / across + 0.5),
        values=crop[keep][order],
        width_px=float(across),
    )
