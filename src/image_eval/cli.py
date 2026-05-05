from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import cv2
import numpy as np

from image_eval.mtf_results import MTFReportPaths, calculate_mtf_report, save_mtf_report
from image_eval.nps_results import NPSReportPaths, calculate_nps_report, save_nps_report
from image_eval.registered_template import project_template_rois
from image_eval.registration import register_subject_in_base
from image_eval.registration_artifacts import (
    RegistrationArtifactPaths,
    save_registration_artifact_plots,
)
from image_eval.template_io import base_image_path, load_2d_npy, load_template


class ImageEvaluationPaths(NamedTuple):
    output_dir: Path
    mtf_paths: MTFReportPaths
    nps_paths: NPSReportPaths
    registration_paths: RegistrationArtifactPaths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-eval",
        description=(
            "Evaluate a .npy image against a template and write registration, MTF, and NPS artifacts."
        ),
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("template_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        evaluate_image(args.image, args.template_json, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"image-eval: error: {error}", file=sys.stderr)
        return 1

    return 0


def evaluate_image(image_path: Path, template_json: Path, output_dir: Path) -> ImageEvaluationPaths:
    template = load_template(template_json)
    subject_image = load_2d_npy(image_path)
    base_path = base_image_path(template_json, template)

    if _same_path(image_path, base_path):
        base_image = subject_image
        registration = _identity_registration(subject_image.shape)
    else:
        base_image = load_2d_npy(base_path)
        registration = {
            "mode": "feature",
            **register_subject_in_base(base_image, subject_image),
        }

    registration = _with_registration_context(
        registration,
        image_path=image_path,
        base_image_path_value=base_path,
        base_shape=base_image.shape,
        subject_shape=subject_image.shape,
    )
    registered_template = project_template_rois(
        template,
        registration["transform_subject_to_base"],
        subject_image.shape,
        image_path.resolve(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    registration_paths = save_registration_artifacts(
        base_image,
        subject_image,
        registration,
        registered_template,
        output_dir / "registration",
    )

    report = calculate_mtf_report(subject_image, registered_template)
    mtf_paths = save_mtf_report(report, output_dir)
    nps_report = calculate_nps_report(subject_image, registered_template)
    nps_paths = save_nps_report(nps_report, output_dir)

    return ImageEvaluationPaths(
        output_dir=output_dir,
        mtf_paths=mtf_paths,
        nps_paths=nps_paths,
        registration_paths=registration_paths,
    )


def save_registration_artifacts(
    base_image: np.ndarray,
    subject_image: np.ndarray,
    registration: dict[str, Any],
    registered_template: dict[str, Any],
    registration_dir: Path,
) -> RegistrationArtifactPaths:
    registration_dir.mkdir(parents=True, exist_ok=True)
    registration_json_path = registration_dir / "registration.json"
    registered_template_path = registration_dir / "registered_template.json"

    with registration_json_path.open("w") as file:
        json.dump(registration, file, indent=2)
        file.write("\n")
    with registered_template_path.open("w") as file:
        json.dump(registered_template, file, indent=2)
        file.write("\n")

    roi_overlay_path, image_overlay_path = save_registration_artifact_plots(
        base_image,
        subject_image,
        registered_template,
        registration["transform_subject_to_base"],
        registration_dir,
    )

    return RegistrationArtifactPaths(
        registration_dir=registration_dir,
        registration_json_path=registration_json_path,
        registered_template_path=registered_template_path,
        roi_overlay_path=roi_overlay_path,
        image_overlay_path=image_overlay_path,
    )


def _identity_registration(subject_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = subject_shape
    transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    corners = [
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)],
    ]
    return {
        "mode": "identity",
        "bbox_xyxy": [0.0, 0.0, float(width - 1), float(height - 1)],
        "corners_xy": corners,
        "transform_subject_to_base": transform,
        "scale_base_per_subject_pixel": 1.0,
        "rotation_degrees": 0.0,
        "ssim_score": None,
        "candidate_count": 0,
        "candidate_source": "identity",
        "good_matches": 0,
        "inlier_matches": 0,
    }


def _with_registration_context(
    registration: dict[str, Any],
    *,
    image_path: Path,
    base_image_path_value: Path,
    base_shape: tuple[int, int],
    subject_shape: tuple[int, int],
) -> dict[str, Any]:
    transform = np.asarray(registration["transform_subject_to_base"], dtype=np.float64)
    if transform.shape != (2, 3):
        raise ValueError("registration transform_subject_to_base must be a 2x3 affine transform")
    transform_base_to_subject = cv2.invertAffineTransform(transform)
    base_height, base_width = base_shape
    subject_height, subject_width = subject_shape

    return {
        **registration,
        "base_image_path": str(base_image_path_value.resolve()),
        "subject_image_path": str(image_path.resolve()),
        "base_image_shape": {
            "height": base_height,
            "width": base_width,
        },
        "subject_image_shape": {
            "height": subject_height,
            "width": subject_width,
        },
        "transform_base_to_subject": transform_base_to_subject.tolist(),
    }


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve(strict=True) == second.resolve(strict=True)
    except OSError:
        return first.resolve() == second.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
