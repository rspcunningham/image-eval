from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from image_eval.registration import register_subject_in_base
from image_eval.template_io import base_image_path, load_2d_npy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a subject .npy image onto the base image from a template JSON."
    )
    parser.add_argument("template_json", type=Path)
    parser.add_argument("subject_image", type=Path)
    args = parser.parse_args(argv)

    try:
        base_image = load_2d_npy(base_image_path(args.template_json))
        subject_image = load_2d_npy(args.subject_image)
        result = register_subject_in_base(base_image, subject_image)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"register-crop: error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
