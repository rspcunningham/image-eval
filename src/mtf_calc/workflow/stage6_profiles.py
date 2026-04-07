from __future__ import annotations

from typing import Any

import numpy as np


Rect = tuple[int, int, int, int]


def _rect_array(source: np.ndarray, rect: Rect) -> np.ndarray:
    row, col, height, width = rect
    return source[row:row + height, col:col + width]


def _rect_tuple(slot: dict[str, Any]) -> Rect:
    rect = slot.get("rect") or {}
    return (
        int(rect["row"]),
        int(rect["col"]),
        int(rect["height"]),
        int(rect["width"]),
    )


def _profile_for_bar_slot(source: np.ndarray, slot: dict[str, Any]) -> dict[str, Any]:
    rect = _rect_tuple(slot)
    roi = _rect_array(source, rect)
    axis = str(slot["axis"])
    if axis == "X":
        profile = roi.mean(axis=0)
        averaged_axis = "Y"
    else:
        profile = roi.mean(axis=1)
        averaged_axis = "X"

    return {
        "key": slot["key"],
        "label": slot["label"],
        "group": int(slot["group"]),
        "roiNumber": int(slot["roiNumber"]),
        "element": int(slot.get("element", slot["roiNumber"])),
        "axis": axis,
        "rect": slot["rect"],
        "spatialFrequencyLpPerMm": float(slot["spatialFrequencyLpPerMm"]),
        "lineWidthUm": float(slot["lineWidthUm"]),
        "profileAxis": axis,
        "averagedAxis": averaged_axis,
        "sampleCount": int(roi.shape[0] if axis == "X" else roi.shape[1]),
        "profileLength": int(profile.shape[0]),
        "crop": {"left": 0, "right": 0},
        "rawProfile": profile.astype(np.float64).tolist(),
    }


def _mean_for_norm_slot(source: np.ndarray, slot: dict[str, Any]) -> dict[str, Any]:
    rect = _rect_tuple(slot)
    roi = _rect_array(source, rect)
    return {
        "key": slot["key"],
        "label": slot["label"],
        "tone": slot["tone"],
        "rect": slot["rect"],
        "mean": float(np.mean(roi, dtype=np.float64)),
        "pixelCount": int(roi.size),
    }


def build_stage6_profiles(
    source: np.ndarray,
    *,
    bar_slots: list[dict[str, Any]],
    norm_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    profiles = [_profile_for_bar_slot(source, slot) for slot in bar_slots]
    normalization_slots = [_mean_for_norm_slot(source, slot) for slot in norm_slots]
    black_slot = next((slot for slot in normalization_slots if slot["tone"] == "black"), None)
    white_slot = next((slot for slot in normalization_slots if slot["tone"] == "white"), None)
    black_mean = black_slot["mean"] if black_slot is not None else None
    white_mean = white_slot["mean"] if white_slot is not None else None
    contrast = (
        float(white_mean - black_mean)
        if black_mean is not None and white_mean is not None
        else None
    )
    normalization_valid = contrast is not None and abs(contrast) > 1e-9

    for profile in profiles:
        raw_profile = np.asarray(profile["rawProfile"], dtype=np.float64)
        profile["normalizedProfile"] = (
            ((raw_profile - black_mean) / contrast).tolist()
            if normalization_valid
            else None
        )
        # Keep a compatibility alias while the UI migrates to explicit profile fields.
        profile["profile"] = profile["rawProfile"]

    return {
        "profiles": profiles,
        "normalization": {
            "slots": normalization_slots,
            "blackMean": black_mean,
            "whiteMean": white_mean,
            "contrast": contrast,
            "normalized": normalization_valid,
        },
    }
