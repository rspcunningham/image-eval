from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, cast

import cv2
import numpy as np


RATIO_TEST = 0.75
REPROJECTION_THRESHOLD = 5.0
MIN_INLIERS = 8

CANDIDATE_KNN = 30
MAX_DESCRIPTOR_DISTANCE = 130.0
HOUGH_BIN_PIXELS = 50.0
HOUGH_BIN_LOG_SCALE = 0.15
HOUGH_BIN_DEGREES = 15.0

MIN_SCALE = 0.1
MAX_SCALE = 1.0


@dataclass(frozen=True)
class RegistrationCandidate:
    transform: np.ndarray
    source: str
    bbox_xyxy: list[float]
    corners_xy: list[list[float]]
    scale: float
    rotation_degrees: float
    good_matches: int
    inlier_matches: int
    ssim_score: float = float("-inf")


def registration_candidates(
    base_image: np.ndarray,
    subject_image: np.ndarray,
) -> list[RegistrationCandidate]:
    detector = cast(Any, cv2).AKAZE_create()
    subject_keypoints, subject_descriptors = detector.detectAndCompute(subject_image, None)
    base_keypoints, base_descriptors = detector.detectAndCompute(base_image, None)

    if base_descriptors is None or subject_descriptors is None:
        raise RuntimeError("could not find enough AKAZE features in one of the images")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates: list[RegistrationCandidate] = []

    ratio_pairs = matcher.knnMatch(subject_descriptors, base_descriptors, k=2)
    ratio_matches = [
        first
        for pair in ratio_pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < RATIO_TEST * second.distance
    ]
    ratio_candidate = _candidate_from_matches(
        base_image.shape,
        subject_image.shape,
        subject_keypoints,
        base_keypoints,
        ratio_matches,
        source="ratio_ransac",
    )
    if ratio_candidate is not None:
        candidates.append(ratio_candidate)

    expanded_pairs = matcher.knnMatch(subject_descriptors, base_descriptors, k=CANDIDATE_KNN)
    for matches in _hough_match_groups(expanded_pairs, subject_keypoints, base_keypoints):
        candidate = _candidate_from_matches(
            base_image.shape,
            subject_image.shape,
            subject_keypoints,
            base_keypoints,
            matches,
            source="hough_ransac",
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def transformed_corners(image_shape: tuple[int, int], transform: np.ndarray) -> np.ndarray:
    height, width = image_shape
    corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.transform(corners, transform).reshape(-1, 2)


def _hough_match_groups(
    match_groups: Sequence[Sequence[Any]],
    subject_keypoints: Sequence[Any],
    base_keypoints: Sequence[Any],
) -> list[Sequence[Any]]:
    bins: dict[tuple[int, int, int, int], list[Any]] = {}

    for matches in match_groups:
        for match in matches:
            if match.distance > MAX_DESCRIPTOR_DISTANCE:
                continue

            subject_keypoint = subject_keypoints[match.queryIdx]
            base_keypoint = base_keypoints[match.trainIdx]
            if subject_keypoint.size <= 0:
                continue

            scale = base_keypoint.size / subject_keypoint.size
            if not MIN_SCALE <= scale <= MAX_SCALE:
                continue

            angle_delta = (base_keypoint.angle - subject_keypoint.angle + 180.0) % 360.0 - 180.0
            angle_radians = math.radians(angle_delta)
            subject_x, subject_y = subject_keypoint.pt
            base_x, base_y = base_keypoint.pt
            cos_angle = math.cos(angle_radians)
            sin_angle = math.sin(angle_radians)
            offset_x = base_x - scale * (cos_angle * subject_x - sin_angle * subject_y)
            offset_y = base_y - scale * (sin_angle * subject_x + cos_angle * subject_y)

            bin_key = (
                round(offset_x / HOUGH_BIN_PIXELS),
                round(offset_y / HOUGH_BIN_PIXELS),
                round(math.log(scale) / HOUGH_BIN_LOG_SCALE),
                round(angle_delta / HOUGH_BIN_DEGREES),
            )
            bins.setdefault(bin_key, []).append(match)

    min_votes = max(MIN_INLIERS, 5)
    return [
        matches
        for _, matches in sorted(bins.items(), key=lambda item: len(item[1]), reverse=True)
        if len(matches) >= min_votes
    ]


def _candidate_from_matches(
    base_shape: tuple[int, int],
    subject_shape: tuple[int, int],
    subject_keypoints: Sequence[Any],
    base_keypoints: Sequence[Any],
    matches: Sequence[Any],
    *,
    source: str,
) -> RegistrationCandidate | None:
    if len(matches) < MIN_INLIERS:
        return None

    subject_points = np.array(
        [subject_keypoints[match.queryIdx].pt for match in matches],
        dtype=np.float32,
    )
    base_points = np.array(
        [base_keypoints[match.trainIdx].pt for match in matches],
        dtype=np.float32,
    )
    transform, inliers = cv2.estimateAffinePartial2D(
        subject_points.reshape(-1, 1, 2),
        base_points.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=REPROJECTION_THRESHOLD,
        maxIters=5000,
        confidence=0.995,
    )
    if transform is None or inliers is None:
        return None

    inlier_count = int(inliers.sum())
    if inlier_count < MIN_INLIERS:
        return None

    scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    if not MIN_SCALE <= scale <= MAX_SCALE:
        return None

    corners = transformed_corners(subject_shape, transform)
    if not _corners_are_in_image(corners, base_shape):
        return None

    return _make_candidate(
        subject_shape,
        transform,
        source=source,
        good_matches=len(matches),
        inlier_matches=inlier_count,
    )


def _make_candidate(
    subject_shape: tuple[int, int],
    transform: np.ndarray,
    *,
    source: str,
    good_matches: int,
    inlier_matches: int,
) -> RegistrationCandidate:
    corners = transformed_corners(subject_shape, transform)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    rotation_degrees = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))

    return RegistrationCandidate(
        transform=transform,
        source=source,
        bbox_xyxy=[float(x0), float(y0), float(x1), float(y1)],
        corners_xy=corners.tolist(),
        scale=scale,
        rotation_degrees=rotation_degrees,
        good_matches=good_matches,
        inlier_matches=inlier_matches,
    )


def _corners_are_in_image(
    corners: np.ndarray,
    image_shape: tuple[int, int],
    *,
    margin: float = 10.0,
) -> bool:
    height, width = image_shape
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return bool(
        x0 >= -margin and y0 >= -margin and x1 <= width - 1 + margin and y1 <= height - 1 + margin
    )
