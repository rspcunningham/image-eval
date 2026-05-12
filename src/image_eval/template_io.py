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


def load_2d_npy(path: Path) -> np.ndarray:
    image = np.load(path)
    if image.ndim != 2:
        raise ValueError(f"{path} is {image.ndim}D; expected a 2D .npy array")
    if np.iscomplexobj(image):
        return np.real(image * np.conj(image))
    return image
