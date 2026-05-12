from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from image_eval.registration import register_subject_in_base
from image_eval.sources import load_image_source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a subject .npy image onto an explicit base .npy image."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--subject-url", required=True)
    args = parser.parse_args(argv)

    try:
        base_image = load_image_source(args.base_url)
        subject_image = load_image_source(args.subject_url)
        result = register_subject_in_base(base_image, subject_image)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"register-crop: error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
