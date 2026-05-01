from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[Any]
DisplayImage = NDArray[np.uint8]


def load_image(path: str | Path) -> ImageArray:
    image_path = Path(path)
    if image_path.suffix.lower() == ".npy":
        return cast(ImageArray, np.load(image_path))

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cast(ImageArray, image)


def image_size(image: ImageArray) -> tuple[int, int]:
    if image.ndim < 2:
        raise ValueError("Expected a 2D grayscale image or a 3D color image.")
    height, width = image.shape[:2]
    return int(width), int(height)


def to_display_image(image: ImageArray) -> DisplayImage:
    if image.ndim == 2:
        gray = _scale_to_uint8(image)
        return cast(DisplayImage, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    if image.ndim != 3:
        raise ValueError("Expected a 2D grayscale image or a 3D color image.")

    channel_count = int(image.shape[2])
    if channel_count == 1:
        gray = _scale_to_uint8(image[:, :, 0])
        return cast(DisplayImage, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    if channel_count == 3:
        return _scale_to_uint8(image)
    if channel_count == 4:
        bgr = image[:, :, :3]
        return _scale_to_uint8(bgr)

    raise ValueError(f"Unsupported image channel count: {channel_count}")


def _scale_to_uint8(image: ImageArray) -> DisplayImage:
    values = np.asarray(image)
    if values.dtype == np.uint8:
        return cast(DisplayImage, values.copy())

    values_float = values.astype(np.float32, copy=False)
    finite = values_float[np.isfinite(values_float)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)

    scaled = (np.clip(values_float, low, high) - low) / (high - low)
    return cast(DisplayImage, np.round(scaled * 255.0).astype(np.uint8))
