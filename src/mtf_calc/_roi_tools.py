from __future__ import annotations

import base64
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Anchor, BarSection, NormRegion, Point, Roi


def build_select_roi_config(
    raw_image: NDArray[np.float32],
    size_ref: Roi | None = None,
    prompt: str | None = None,
) -> dict[str, object]:
    return {
        "tool": "select-roi",
        "rows": cast(int, raw_image.shape[0]),
        "cols": cast(int, raw_image.shape[1]),
        "rawImage": _encode_raw_image(raw_image),
        "sizeRef": _serialize_size_ref(size_ref),
        "prompt": prompt,
    }


def build_show_anchor_config(raw_image: NDArray[np.float32], anchor: Anchor) -> dict[str, object]:
    return {
        "tool": "show-anchor",
        "rows": cast(int, raw_image.shape[0]),
        "cols": cast(int, raw_image.shape[1]),
        "rawImage": _encode_raw_image(raw_image),
        "anchor": {
            "roi": roi_to_payload(anchor.roi),
            "centroid": {
                "x": float(anchor.centroid.x),
                "y": float(anchor.centroid.y),
            },
        },
    }


def build_show_rois_config(
    raw_image: NDArray[np.float32],
    *,
    anchor: Anchor,
    norm_rois: dict[NormRegion, Roi],
    bar_rois: dict[BarSection, Roi],
) -> dict[str, object]:
    return {
        "tool": "show-rois",
        "rows": cast(int, raw_image.shape[0]),
        "cols": cast(int, raw_image.shape[1]),
        "rawImage": _encode_raw_image(raw_image),
        "anchor": {
            "roi": roi_to_payload(anchor.roi),
            "centroid": {
                "x": float(anchor.centroid.x),
                "y": float(anchor.centroid.y),
            },
        },
        "normRois": [
            {
                "label": "Black norm" if region == 0 else "White norm",
                "kind": "norm",
                "roi": roi_to_payload(roi),
            }
            for region, roi in sorted(norm_rois.items())
        ],
        "barRois": [
            {
                "label": f"G{section.group} E{section.element} {section.dim}",
                "kind": "bar",
                "roi": roi_to_payload(roi),
            }
            for section, roi in sorted(
                bar_rois.items(),
                key=lambda item: (item[0].group, item[0].element, item[0].dim),
            )
        ],
    }


def roi_from_payload(payload: dict[str, object]) -> Roi:
    left = _as_float(payload.get("left"))
    top = _as_float(payload.get("top"))
    right = _as_float(payload.get("right"))
    bottom = _as_float(payload.get("bottom"))

    if right - left < 2.0 or bottom - top < 2.0:
        raise RuntimeError("ROI must be at least 2x2 pixels")

    return Roi(
        top_left=Point(x=left, y=top),
        top_right=Point(x=right, y=top),
        bottom_left=Point(x=left, y=bottom),
        bottom_right=Point(x=right, y=bottom),
    )


def roi_to_payload(roi: Roi) -> dict[str, float]:
    return {
        "left": float(roi.top_left.x),
        "top": float(roi.top_left.y),
        "right": float(roi.top_right.x),
        "bottom": float(roi.bottom_left.y),
    }


def _encode_raw_image(raw_image: NDArray[np.float32]) -> dict[str, object]:
    raw = np.ascontiguousarray(raw_image, dtype=np.float32)
    encoded = base64.b64encode(raw.tobytes(order="C")).decode("ascii")
    return {
        "dtype": "float32",
        "rows": cast(int, raw.shape[0]),
        "cols": cast(int, raw.shape[1]),
        "data": encoded,
    }


def _serialize_size_ref(size_ref: Roi | None) -> dict[str, float] | None:
    if size_ref is None:
        return None

    width = max(
        abs(size_ref.top_right.x - size_ref.top_left.x),
        abs(size_ref.bottom_right.x - size_ref.bottom_left.x),
    )
    height = max(
        abs(size_ref.bottom_left.y - size_ref.top_left.y),
        abs(size_ref.bottom_right.y - size_ref.top_right.y),
    )
    return {
        "width": float(width),
        "height": float(height),
    }


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected numeric ROI bounds")
    return float(value)
