from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import patches, pyplot as plt  # noqa: E402


class RegistrationArtifactPaths(NamedTuple):
    registration_dir: Path
    registration_json_path: Path
    registered_template_path: Path
    roi_overlay_path: Path
    image_overlay_path: Path


def save_registration_artifact_plots(
    base_image: np.ndarray,
    subject_image: np.ndarray,
    registered_template: dict[str, Any],
    transform_subject_to_base: Sequence[Sequence[float]],
    registration_dir: Path,
) -> tuple[Path, Path]:
    registration_dir.mkdir(parents=True, exist_ok=True)
    roi_overlay_path = registration_dir / "projected_rois.png"
    image_overlay_path = registration_dir / "overlay.png"
    save_projected_rois_plot(subject_image, registered_template, roi_overlay_path)
    save_registration_overlay_plot(
        base_image,
        subject_image,
        transform_subject_to_base,
        image_overlay_path,
    )
    return roi_overlay_path, image_overlay_path


def save_projected_rois_plot(
    subject_image: np.ndarray,
    registered_template: dict[str, Any],
    output_path: Path,
) -> None:
    image = _display_uint8(subject_image)
    figure, axis = _image_figure(image)
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)

    normalization_rois = registered_template["normalization_rois"]
    _draw_rect(axis, normalization_rois["black"], "black", "#e15759", linewidth=2.0)
    _draw_rect(axis, normalization_rois["white"], "white", "#59a14f", linewidth=2.0)

    for roi in registered_template["bar_rois"]:
        orientation = roi["orientation"]
        color = "#4e79a7" if orientation == "X" else "#f28e2b"
        label = f"G{roi['group']} E{roi['element']} {orientation}"
        _draw_rect(axis, roi["rect"], label, color, linewidth=1.2)

    axis.set_title("Projected ROIs on Subject Image")
    axis.set_axis_off()
    figure.tight_layout(pad=0.2)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0.03)
    finally:
        plt.close(figure)


def save_registration_overlay_plot(
    base_image: np.ndarray,
    subject_image: np.ndarray,
    transform_subject_to_base: Sequence[Sequence[float]],
    output_path: Path,
) -> None:
    subject_display = _display_uint8(subject_image)
    base_display = _display_uint8(base_image)
    subject_height, subject_width = subject_display.shape

    transform = np.asarray(transform_subject_to_base, dtype=np.float64)
    if transform.shape != (2, 3):
        raise ValueError("transform_subject_to_base must be a 2x3 affine transform")
    transform_base_to_subject = cv2.invertAffineTransform(transform)

    warped_base = cv2.warpAffine(
        base_display,
        transform_base_to_subject,
        dsize=(subject_width, subject_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    base_mask = cv2.warpAffine(
        np.full(base_display.shape, 255, dtype=np.uint8),
        transform_base_to_subject,
        dsize=(subject_width, subject_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    overlay = np.zeros((subject_height, subject_width, 3), dtype=np.uint8)
    overlay[..., 0] = np.where(base_mask > 0, warped_base, 0)
    overlay[..., 1] = subject_display
    overlay[..., 2] = subject_display

    figure, axis = _image_figure(subject_display)
    axis.imshow(overlay)
    axis.set_title("Registration Overlay: base red, subject cyan")
    axis.set_axis_off()
    figure.tight_layout(pad=0.2)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0.03)
    finally:
        plt.close(figure)


def _display_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image is {image.ndim}D; expected a 2D array")

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    fill_value = float(np.median(image[finite]))
    image = np.where(finite, image, fill_value)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def _image_figure(image: np.ndarray):
    height, width = image.shape
    long_side = max(width, height)
    scale = 8.0 / long_side if long_side else 1.0
    figure_width = max(4.0, width * scale)
    figure_height = max(4.0, height * scale)
    return plt.subplots(figsize=(figure_width, figure_height))


def _draw_rect(axis: Any, rect: Any, label: str, color: str, *, linewidth: float) -> None:
    if not isinstance(rect, dict):
        return
    required = ("x0", "y0", "x1", "y1")
    if any(not isinstance(rect.get(key), int) for key in required):
        return

    x0 = rect["x0"]
    y0 = rect["y0"]
    width = rect["x1"] - rect["x0"]
    height = rect["y1"] - rect["y0"]
    if width <= 0 or height <= 0:
        return

    axis.add_patch(
        patches.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )
    axis.text(
        x0,
        max(0, y0 - 3),
        label,
        color=color,
        fontsize=6,
        fontweight="bold",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1},
    )
