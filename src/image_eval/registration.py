from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from image_eval.registration_candidates import (
    RegistrationCandidate,
    registration_candidates,
    transformed_corners,
)


def register_subject_in_base(base_image: np.ndarray, subject_image: np.ndarray) -> dict[str, Any]:
    """Locate subject_image inside base_image using SSIM-ranked feature candidates.

    The returned transform maps x/y coordinates in subject_image into x/y coordinates
    in base_image.
    """
    prepared_base = _preprocess(base_image)
    prepared_subject = _preprocess(subject_image)

    candidates = registration_candidates(prepared_base, prepared_subject)
    if not candidates:
        raise RuntimeError("could not generate any valid registration candidates")

    scored_candidates = [
        replace(
            candidate,
            ssim_score=_masked_ssim_for_transform(
                prepared_base,
                prepared_subject,
                candidate.transform,
            ),
        )
        for candidate in candidates
    ]
    best = max(scored_candidates, key=lambda candidate: candidate.ssim_score)
    return _candidate_result(best, candidate_count=len(scored_candidates))


def _candidate_result(candidate: RegistrationCandidate, *, candidate_count: int) -> dict[str, Any]:
    return {
        "bbox_xyxy": candidate.bbox_xyxy,
        "corners_xy": candidate.corners_xy,
        "transform_subject_to_base": candidate.transform.tolist(),
        "scale_base_per_subject_pixel": candidate.scale,
        "rotation_degrees": candidate.rotation_degrees,
        "ssim_score": candidate.ssim_score,
        "candidate_count": candidate_count,
        "candidate_source": candidate.source,
        "good_matches": candidate.good_matches,
        "inlier_matches": candidate.inlier_matches,
    }


def _masked_ssim_for_transform(
    base_image: np.ndarray,
    subject_image: np.ndarray,
    transform: np.ndarray,
) -> float:
    base_height, base_width = base_image.shape
    subject_height, subject_width = subject_image.shape
    warped_subject = cv2.warpAffine(
        subject_image,
        transform,
        dsize=(base_width, base_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    subject_mask = cv2.warpAffine(
        np.full((subject_height, subject_width), 255, dtype=np.uint8),
        transform,
        dsize=(base_width, base_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    corners = transformed_corners(subject_image.shape, transform)
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(base_width, x1 + 1)
    y1 = min(base_height, y1 + 1)
    if x1 <= x0 or y1 <= y0:
        return float("-inf")

    mask_crop = subject_mask[y0:y1, x0:x1] > 0
    if int(mask_crop.sum()) < 64:
        return float("-inf")

    base_crop = base_image[y0:y1, x0:x1].astype(np.float32) / 255.0
    subject_crop = warped_subject[y0:y1, x0:x1].astype(np.float32) / 255.0

    sigma = 1.5
    mean_base = cv2.GaussianBlur(base_crop, (0, 0), sigma)
    mean_subject = cv2.GaussianBlur(subject_crop, (0, 0), sigma)
    mean_base2 = mean_base * mean_base
    mean_subject2 = mean_subject * mean_subject
    mean_cross = mean_base * mean_subject
    var_base = np.maximum(cv2.GaussianBlur(base_crop * base_crop, (0, 0), sigma) - mean_base2, 0.0)
    var_subject = np.maximum(
        cv2.GaussianBlur(subject_crop * subject_crop, (0, 0), sigma) - mean_subject2,
        0.0,
    )
    cov = cv2.GaussianBlur(base_crop * subject_crop, (0, 0), sigma) - mean_cross

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mean_cross + c1) * (2.0 * cov + c2)) / (
        (mean_base2 + mean_subject2 + c1) * (var_base + var_subject + c2) + 1e-12
    )
    return float(np.mean(np.clip(ssim_map, -1.0, 1.0)[mask_crop]))


def _preprocess(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("image contains no finite values")

    fill_value = float(np.median(image[finite]))
    image = np.where(finite, image, fill_value)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    stretched = np.clip((image - low) / (high - low), 0, 1)
    image8 = (stretched * 255).astype(np.uint8)
    return cv2.GaussianBlur(image8, (0, 0), 1.0)
