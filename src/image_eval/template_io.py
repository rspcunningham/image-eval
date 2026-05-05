from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_template(template_json: Path) -> dict[str, Any]:
    with template_json.open() as file:
        template = json.load(file)
    if not isinstance(template, dict):
        raise ValueError(f"{template_json} must contain a JSON object")
    return template


def base_image_path(template_json: Path, template: dict[str, Any] | None = None) -> Path:
    template = template if template is not None else load_template(template_json)
    raw_path = template.get("base_image_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{template_json} does not contain a string base_image_path")

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return template_json.parent / path


def load_2d_npy(path: Path) -> np.ndarray:
    image = np.load(path)
    if image.ndim != 2:
        raise ValueError(f"{path} is {image.ndim}D; expected a 2D .npy array")
    if np.iscomplexobj(image):
        return np.real(image * np.conj(image))
    return image
