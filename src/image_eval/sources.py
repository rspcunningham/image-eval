from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np

from image_eval.template_io import load_2d_npy


def load_image_source(source: str | Path) -> np.ndarray:
    data = _source_bytes(source)
    if data is None:
        return load_2d_npy(Path(source))

    image = np.load(BytesIO(data))
    return _as_2d_image(image, str(source))


def load_template_source(source: str | Path) -> dict[str, Any]:
    data = _source_bytes(source)
    if data is None:
        with Path(source).open() as file:
            template = json.load(file)
    else:
        template = json.loads(data.decode("utf-8"))

    if not isinstance(template, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return template


def _source_bytes(source: str | Path) -> bytes | None:
    parsed = urlparse(str(source))
    if parsed.scheme in ("http", "https"):
        with urlopen(str(source), timeout=30) as response:
            return response.read()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()
    if parsed.scheme:
        raise ValueError(f"unsupported source URL scheme: {parsed.scheme}")
    return None


def _as_2d_image(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"{label} is {image.ndim}D; expected a 2D .npy array")
    if np.iscomplexobj(image):
        return np.real(image * np.conj(image))
    return image
