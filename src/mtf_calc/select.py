# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import base64
import io
from pathlib import Path
from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import webview

from mtf_calc.models import Point, Roi


def select_roi(raw_image: NDArray[np.float32], size_ref: Roi | None = None) -> Roi:
    api = _RoiPickerApi(raw_image=raw_image, size_ref=size_ref)
    ui_path = Path(__file__).with_name("ui") / "roi_picker.html"
    window = webview.create_window(
        title="Select ROI",
        # Serve the local UI through pywebview's built-in HTTP server instead of
        # loading it over file://, which is not well supported by the JS bridge.
        url=str(ui_path.resolve()),
        width=1440,
        height=920,
        min_size=(960, 640),
    )
    if window is None:
        raise RuntimeError("Failed to create ROI picker window")
    api.attach_window(window)
    _register_bridge(window, api)
    webview.start(debug=False)

    roi = api.selected_roi
    if roi is None:
        raise RuntimeError("ROI selection cancelled")
    return roi


class _RoiPickerApi:
    def __init__(self, *, raw_image: NDArray[np.float32], size_ref: Roi | None) -> None:
        self._config: dict[str, object] = {
            "rows": cast(int, raw_image.shape[0]),
            "cols": cast(int, raw_image.shape[1]),
            "imageDataUrl": _encode_image(raw_image),
            "sizeRef": _serialize_size_ref(size_ref),
        }
        self.selected_roi: Roi | None = None
        self._window: webview.Window | None = None

    def attach_window(self, window: webview.Window) -> None:
        self._window = window

    def load_config(self) -> Mapping[str, object]:
        return self._config

    def submit_selection(self, payload: dict[str, object]) -> None:
        self.selected_roi = _roi_from_payload(payload)
        if self._window is not None:
            self._window.destroy()

    def cancel(self) -> None:
        self.selected_roi = None
        if self._window is not None:
            self._window.destroy()


def _register_bridge(window: webview.Window, api: "_RoiPickerApi") -> None:
    # `js_api=` produced an empty `window.pywebview.api` on this Cocoa setup,
    # while `window.expose(...)` exports the functions correctly.
    window.expose(
        api.load_config,
        api.submit_selection,
        api.cancel,
    )


def _encode_image(raw_image: NDArray[np.float32]) -> str:
    clipped = np.clip(raw_image, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def _roi_from_payload(payload: dict[str, object]) -> Roi:
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


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected numeric ROI bounds")
    return float(value)
