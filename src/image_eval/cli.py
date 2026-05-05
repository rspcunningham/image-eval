import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "initialize":
        return _run_initialize(args)
    if args.command == "register":
        return _run_python_phase("register.py", args.args)
    if args.command == "evaluate":
        return _run_python_phase("evaluate.py", args.args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-eval",
        description="Run image-eval workflow phases from one CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "initialize",
        help="Open the native macOS ROI selector and write a template JSON file.",
    )
    initialize.add_argument("source_image_path", type=Path)
    initialize.add_argument("template_path", type=Path)
    initialize.add_argument(
        "--groups",
        metavar="SPEC",
        help="Group selection, such as 4-7 or 4,5,7. Required for new templates.",
    )
    initialize.add_argument(
        "--elements",
        metavar="SPEC",
        help="Element selection, such as 1-6 or 1,3,6. Required for new templates.",
    )

    register = subparsers.add_parser(
        "register",
        help="Run the registration phase.",
    )
    register.add_argument("args", nargs=argparse.REMAINDER)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run the evaluation phase.",
    )
    evaluate.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def _run_initialize(args: argparse.Namespace) -> int:
    package_path = _repo_root() / "native" / "ROISelector"
    executable = _built_roi_selector(package_path)
    if executable is None:
        command = [
            "swift",
            "run",
            "--package-path",
            str(package_path),
            "ROISelector",
            str(args.source_image_path),
            str(args.template_path),
        ]
    else:
        command = [
            str(executable),
            str(args.source_image_path),
            str(args.template_path),
        ]
    if args.groups is not None:
        command.extend(["--groups", args.groups])
    if args.elements is not None:
        command.extend(["--elements", args.elements])
    return _run(command)


def _run_python_phase(script_name: str, extra_args: list[str]) -> int:
    script_path = _repo_root() / script_name
    return _run([sys.executable, str(script_path), *extra_args])


def _run(command: list[str]) -> int:
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as error:
        print(f"image-eval: error: missing executable: {error.filename}", file=sys.stderr)
        return 127
    return completed.returncode


def _built_roi_selector(package_path: Path) -> Path | None:
    configuration = os.environ.get("IMAGE_EVAL_SWIFT_CONFIGURATION", "debug")
    candidates = [
        package_path / ".build" / configuration / "ROISelector",
        package_path / ".build" / "debug" / "ROISelector",
        package_path / ".build" / "release" / "ROISelector",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
