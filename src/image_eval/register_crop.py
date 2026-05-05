from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from image_eval.registration import register_subject_in_base


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a subject .npy image onto the base image from a template JSON."
    )
    parser.add_argument("template_json", type=Path)
    parser.add_argument("subject_image", type=Path)
    args = parser.parse_args(argv)

    try:
        base_image = _load_2d_npy(_base_image_path(args.template_json))
        subject_image = _load_2d_npy(args.subject_image)
        result = register_subject_in_base(base_image, subject_image)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"register-crop: error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


def _base_image_path(template_json: Path) -> Path:
    template = _load_template(template_json)
    raw_path = template.get("base_image_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{template_json} does not contain a string base_image_path")

    base_image_path = Path(raw_path)
    if base_image_path.is_absolute():
        return base_image_path
    return template_json.parent / base_image_path


def _load_template(template_json: Path) -> dict[str, Any]:
    with template_json.open() as file:
        template = json.load(file)
    if not isinstance(template, dict):
        raise ValueError(f"{template_json} must contain a JSON object")
    return template


def _load_2d_npy(path: Path) -> np.ndarray:
    image = np.load(path)
    if image.ndim != 2:
        raise ValueError(f"{path} is {image.ndim}D; expected a 2D .npy array")
    return image


if __name__ == "__main__":
    raise SystemExit(main())
