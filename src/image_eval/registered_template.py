from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import cv2
import numpy as np

from image_eval.registration import register_subject_in_base
from image_eval.template_io import base_image_path, load_2d_npy, load_template


Rect = dict[str, int]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project visible template ROIs into a registered subject .npy image."
    )
    parser.add_argument("template_json", type=Path)
    parser.add_argument("subject_image", type=Path)
    parser.add_argument("output_template_json", type=Path)
    args = parser.parse_args(argv)

    try:
        template = load_template(args.template_json)
        base_image = load_2d_npy(base_image_path(args.template_json, template))
        subject_image = load_2d_npy(args.subject_image)
        registration = register_subject_in_base(base_image, subject_image)
        output_template = project_template_rois(
            template,
            registration["transform_subject_to_base"],
            subject_image.shape,
            args.subject_image.resolve(),
        )
        args.output_template_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_template_json.open("w") as file:
            json.dump(output_template, file, indent=2)
            file.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"registered-template: error: {error}", file=sys.stderr)
        return 1

    return 0


def project_template_rois(
    template: dict[str, Any],
    transform_subject_to_base: Any,
    subject_shape: tuple[int, int],
    subject_path: Path,
) -> dict[str, Any]:
    subject_height, subject_width = subject_shape
    transform_base_to_subject = cv2.invertAffineTransform(
        _as_affine_transform(transform_subject_to_base, "transform_subject_to_base")
    )

    return {
        "base_image_path": str(subject_path),
        "source_image": {
            "path": str(subject_path),
            "width": subject_width,
            "height": subject_height,
        },
        "normalization_rois": _project_normalization_rois(
            template["normalization_rois"],
            transform_base_to_subject,
            subject_width,
            subject_height,
        ),
        "bar_rois": _project_bar_rois(
            template["bar_rois"],
            transform_base_to_subject,
            subject_width,
            subject_height,
        ),
    }


def _project_normalization_rois(
    normalization_rois: Any,
    transform_base_to_subject: np.ndarray,
    subject_width: int,
    subject_height: int,
) -> dict[str, Rect | None]:
    return {
        "black": _project_rect_if_visible(
            normalization_rois["black"],
            transform_base_to_subject,
            subject_width,
            subject_height,
        ),
        "white": _project_rect_if_visible(
            normalization_rois["white"],
            transform_base_to_subject,
            subject_width,
            subject_height,
        ),
    }


def _project_bar_rois(
    bar_rois: Any,
    transform_base_to_subject: np.ndarray,
    subject_width: int,
    subject_height: int,
) -> list[dict[str, Any]]:
    projected_rois: list[dict[str, Any]] = []
    for roi in cast(list[dict[str, Any]], bar_rois):
        projected_rect = _project_rect_if_visible(
            roi["rect"],
            transform_base_to_subject,
            subject_width,
            subject_height,
        )
        if projected_rect is None:
            continue

        projected_rois.append({
            "group": roi["group"],
            "element": roi["element"],
            "orientation": roi["orientation"],
            "rect": projected_rect,
        })

    return projected_rois


def _project_rect_if_visible(
    rect: Any,
    transform_base_to_subject: np.ndarray,
    subject_width: int,
    subject_height: int,
) -> Rect | None:
    if rect is None:
        return None
    source_rect = _as_rect(rect)
    corners = np.array(
        [
            [source_rect["x0"], source_rect["y0"]],
            [source_rect["x1"], source_rect["y0"]],
            [source_rect["x1"], source_rect["y1"]],
            [source_rect["x0"], source_rect["y1"]],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, transform_base_to_subject).reshape(-1, 2)

    x0, y0 = transformed.min(axis=0)
    x1, y1 = transformed.max(axis=0)
    projected_rect = {
        "x0": math.floor(float(x0)),
        "y0": math.floor(float(y0)),
        "x1": math.ceil(float(x1)),
        "y1": math.ceil(float(y1)),
    }

    if not _rect_is_fully_visible(projected_rect, subject_width, subject_height):
        return None
    return projected_rect


def _rect_is_fully_visible(rect: Rect, image_width: int, image_height: int) -> bool:
    return (
        rect["x0"] >= 0
        and rect["y0"] >= 0
        and rect["x1"] <= image_width
        and rect["y1"] <= image_height
        and rect["x1"] > rect["x0"]
        and rect["y1"] > rect["y0"]
    )


def _as_rect(rect: Any) -> Rect:
    if not isinstance(rect, dict):
        raise ValueError("ROI rect must be a JSON object or null")

    projected: Rect = {}
    for key in ("x0", "y0", "x1", "y1"):
        value = rect.get(key)
        if not isinstance(value, int):
            raise ValueError(f"ROI rect {key} must be an integer")
        projected[key] = value

    if projected["x1"] <= projected["x0"] or projected["y1"] <= projected["y0"]:
        raise ValueError("ROI rect must have positive width and height")
    return projected


def _as_affine_transform(transform: Any, label: str) -> np.ndarray:
    array = np.asarray(transform, dtype=np.float64)
    if array.shape != (2, 3):
        raise ValueError(f"{label} must be a 2x3 affine transform")
    return array


if __name__ == "__main__":
    raise SystemExit(main())
