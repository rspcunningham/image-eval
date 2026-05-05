from __future__ import annotations

from typing import Any, cast

import numpy as np


Rect = dict[str, int]


def as_2d_float_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image is {image.ndim}D; expected a 2D array")
    return image


def finite_roi_pixels(image: np.ndarray, rect: Rect, label: str) -> np.ndarray:
    crop = crop_image(image, rect, label)
    finite_pixels = crop[np.isfinite(crop)]
    if finite_pixels.size == 0:
        raise ValueError(f"{label} contains no finite pixels")
    return finite_pixels


def finite_crop_image(image: np.ndarray, rect: Rect, label: str) -> np.ndarray:
    crop = crop_image(image, rect, label)
    if not np.all(np.isfinite(crop)):
        raise ValueError(f"{label} must contain only finite pixels")
    return crop


def crop_image(image: np.ndarray, rect: Rect, label: str) -> np.ndarray:
    height, width = image.shape
    if rect["x0"] < 0 or rect["y0"] < 0 or rect["x1"] > width or rect["y1"] > height:
        raise ValueError(f"{label} extends outside image bounds")
    return image[rect["y0"] : rect["y1"], rect["x0"] : rect["x1"]]


def as_rect(rect: Any, label: str) -> Rect:
    if not isinstance(rect, dict):
        raise ValueError(f"{label} must be a JSON object")
    rect = cast(dict[str, Any], rect)

    parsed = {
        "x0": as_int(rect.get("x0"), f"{label}.x0"),
        "y0": as_int(rect.get("y0"), f"{label}.y0"),
        "x1": as_int(rect.get("x1"), f"{label}.x1"),
        "y1": as_int(rect.get("y1"), f"{label}.y1"),
    }
    if parsed["x1"] <= parsed["x0"] or parsed["y1"] <= parsed["y0"]:
        raise ValueError(f"{label} must have positive width and height")
    return parsed


def as_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value
