from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import cv2
import numpy as np


def register_crop(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    ratio: float = 0.75,
    reprojection_threshold: float = 5.0,
    min_inliers: int = 8,
) -> dict[str, Any]:
    """Locate the crop represented by image B inside image A.

    The returned transform maps x/y coordinates in B into x/y coordinates in A.
    """
    if image_a.ndim != 2 or image_b.ndim != 2:
        raise ValueError("image_a and image_b must both be 2D arrays")

    a8 = _preprocess(image_a)
    b8 = _preprocess(image_b)

    detector = cast(Any, cv2).AKAZE_create()
    keypoints_b, descriptors_b = detector.detectAndCompute(b8, None)
    keypoints_a, descriptors_a = detector.detectAndCompute(a8, None)

    if descriptors_a is None or descriptors_b is None:
        raise RuntimeError("could not find enough AKAZE features in one of the images")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]

    if len(good_matches) < min_inliers:
        raise RuntimeError(f"only found {len(good_matches)} good feature matches")

    points_b = np.array([keypoints_b[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    points_a = np.array([keypoints_a[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    points_b = points_b.reshape(-1, 1, 2)
    points_a = points_a.reshape(-1, 1, 2)

    transform, inliers = cv2.estimateAffinePartial2D(
        points_b,
        points_a,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=5000,
        confidence=0.995,
    )
    if transform is None or inliers is None:
        raise RuntimeError("could not estimate an affine transform from B into A")

    inlier_count = int(inliers.sum())
    if inlier_count < min_inliers:
        raise RuntimeError(f"only found {inlier_count} RANSAC inlier matches")

    height_b, width_b = image_b.shape
    corners_b = np.array(
        [
            [0, 0],
            [width_b - 1, 0],
            [width_b - 1, height_b - 1],
            [0, height_b - 1],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    corners_a = cv2.transform(corners_b, transform).reshape(-1, 2)

    x0, y0 = corners_a.min(axis=0)
    x1, y1 = corners_a.max(axis=0)

    scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    rotation_degrees = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))

    return {
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "corners_xy": corners_a.tolist(),
        "transform_b_to_a": transform.tolist(),
        "scale_a_per_b_pixel": scale,
        "rotation_degrees": rotation_degrees,
        "good_matches": len(good_matches),
        "inlier_matches": inlier_count,
    }


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


def _load_2d_npy(path: Path) -> np.ndarray:
    image = np.load(path)
    if image.ndim != 2:
        raise ValueError(f"{path} is {image.ndim}D; expected a 2D .npy array")
    return image


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register a 2D .npy crop B onto a larger image A.")
    parser.add_argument("image_a", type=Path)
    parser.add_argument("image_b", type=Path)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--reprojection-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        result = register_crop(
            _load_2d_npy(args.image_a),
            _load_2d_npy(args.image_b),
            ratio=args.ratio,
            reprojection_threshold=args.reprojection_threshold,
            min_inliers=args.min_inliers,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"register-crop: error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
