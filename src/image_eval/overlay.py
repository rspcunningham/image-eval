from typing import Any

import cv2
import numpy as np

from image_eval.images import ImageArray, to_display_image
from image_eval.template_io import iter_template_rects


def show_template_overlay(image: ImageArray, template: dict[str, Any]) -> None:
    display = to_display_image(image)
    overlay = display.copy()

    for label, rect in iter_template_rects(template):
        left = int(round(rect.left))
        top = int(round(rect.top))
        right = int(round(rect.right))
        bottom = int(round(rect.bottom))
        cv2.rectangle(overlay, (left, top), (right, bottom), (80, 220, 80), 2)
        cv2.putText(
            overlay,
            label,
            (left, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 220, 80),
            1,
            cv2.LINE_AA,
        )

    preview = _fit_for_preview(overlay, max_width=1400, max_height=900)
    cv2.imshow("image-eval template preview", preview)
    _ = cv2.waitKey(0)
    cv2.destroyWindow("image-eval template preview")


def _fit_for_preview(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
