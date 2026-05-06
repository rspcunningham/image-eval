from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-eval init",
        description="Create or edit an ROI template.",
    )
    add_init_arguments(parser)
    args = parser.parse_args(argv)
    return run_init(args)


def add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source_npy", help="Source .npy image path.")
    parser.add_argument("template_json", help="ROI template JSON path.")
    parser.add_argument("--groups", help="Group selection, such as 4-7 or 4,5,7.")
    parser.add_argument("--elements", help="Element selection, such as 1-6 or 1,3,6.")


def run_init(args: argparse.Namespace) -> int:
    try:
        with as_file(_roi_selector_binary()) as binary:
            _ensure_executable(binary)
            command = [
                str(binary),
                args.source_npy,
                args.template_json,
            ]
            if args.groups is not None:
                command.extend(["--groups", args.groups])
            if args.elements is not None:
                command.extend(["--elements", args.elements])
            completed = subprocess.run(command, check=False)
    except OSError as error:
        print(f"image-eval init: error: {error}", file=sys.stderr)
        return 127
    return completed.returncode


def _roi_selector_binary():
    binary = files("image_eval").joinpath("_bin", "ROISelector")
    if not binary.is_file():
        raise FileNotFoundError(
            "embedded ROISelector executable not found; reinstall image-eval from a built wheel"
        )
    return binary


def _ensure_executable(path: Path) -> None:
    if os.access(path, os.X_OK):
        return
    path.chmod(path.stat().st_mode | 0o755)


if __name__ == "__main__":
    raise SystemExit(main())
