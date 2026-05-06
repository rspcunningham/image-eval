from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from image_eval.dqe_results import DQEReport, calculate_dqe_report
from image_eval.mtf_results import MTFReport, average_pixels_per_mm_from_fits, calculate_mtf_report
from image_eval.nps_results import NPSReport, SpatialFrequencyCalibration, calculate_nps_report
from image_eval.registered_template import project_template_rois
from image_eval.registration import register_subject_in_base


@dataclass(frozen=True)
class EvaluationResult:
    base_image: np.ndarray
    subject_image: np.ndarray
    registration: dict[str, Any]
    registered_template: dict[str, Any]
    mtf_report: MTFReport
    nps_report: NPSReport
    dqe_report: DQEReport


def evaluate_image(
    *,
    base_image: np.ndarray,
    template: dict[str, Any],
    subject_image: np.ndarray,
) -> EvaluationResult:
    _validate_template_source_image(template, base_image.shape)

    if _same_image(base_image, subject_image):
        registration = _identity_registration(subject_image.shape)
    else:
        registration = {
            "mode": "feature",
            **register_subject_in_base(base_image, subject_image),
        }

    registration = _with_registration_context(
        registration,
        base_shape=base_image.shape,
        subject_shape=subject_image.shape,
    )
    registered_template = project_template_rois(
        template,
        registration["transform_subject_to_base"],
        subject_image.shape,
    )

    mtf_report = calculate_mtf_report(subject_image, registered_template)
    pixels_per_mm = average_pixels_per_mm_from_fits(mtf_report.fitted_profiles)
    nps_report = calculate_nps_report(
        subject_image,
        registered_template,
        frequency_calibration=SpatialFrequencyCalibration(
            unit="lp/mm",
            cycles_per_pixel_multiplier=pixels_per_mm,
        ),
    )
    dqe_report = calculate_dqe_report(mtf_report.results, nps_report.results)

    return EvaluationResult(
        base_image=base_image,
        subject_image=subject_image,
        registration=registration,
        registered_template=registered_template,
        mtf_report=mtf_report,
        nps_report=nps_report,
        dqe_report=dqe_report,
    )


def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    base_height, base_width = result.base_image.shape
    subject_height, subject_width = result.subject_image.shape
    return _json_clean({
        "schema_version": 1,
        "image_shapes": {
            "base": {"height": base_height, "width": base_width},
            "subject": {"height": subject_height, "width": subject_width},
        },
        "registration": result.registration,
        "registered_template": result.registered_template,
        "mtf": {
            "frequency_unit": "lp/mm",
            "rows": [
                {
                    "frequency": row.frequency_lp_per_mm,
                    "x_mtf": row.x_mtf,
                    "y_mtf": row.y_mtf,
                    "average_mtf": row.average_mtf,
                }
                for row in result.mtf_report.results
            ],
        },
        "nps": {
            "frequency_unit": result.nps_report.frequency_calibration.unit,
            "rows": [
                {
                    "frequency": row.frequency,
                    "black_nps": row.black_nps,
                    "white_nps": row.white_nps,
                    "average_nps": row.average_nps,
                }
                for row in result.nps_report.results
            ],
        },
        "dqe": {
            "frequency_unit": "lp/mm",
            "rows": [
                {
                    "frequency": row.frequency_lp_per_mm,
                    "average_mtf": row.average_mtf,
                    "average_nps": row.average_nps,
                    "dqe": row.dqe,
                }
                for row in result.dqe_report.results
            ],
        },
    })


def _validate_template_source_image(template: dict[str, Any], base_shape: tuple[int, int]) -> None:
    source_image = template.get("source_image")
    if not isinstance(source_image, dict):
        raise ValueError("template must contain source_image with width and height")

    width = source_image.get("width")
    height = source_image.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("template source_image width and height must be integers")

    base_height, base_width = base_shape
    if width != base_width or height != base_height:
        raise ValueError(
            "template source_image dimensions "
            f"{width}x{height} do not match base image {base_width}x{base_height}"
        )


def _same_image(base_image: np.ndarray, subject_image: np.ndarray) -> bool:
    return base_image is subject_image or (
        base_image.shape == subject_image.shape
        and np.array_equal(base_image, subject_image, equal_nan=True)
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


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_float_or_none(float(value))
    if isinstance(value, float):
        return _finite_float_or_none(value)
    return value


def _finite_float_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return value
