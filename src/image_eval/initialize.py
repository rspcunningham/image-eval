import subprocess
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    package_path = _repo_root() / "native" / "ROISelector"
    if not package_path.exists():
        print(f"image-eval: error: Swift package not found: {package_path}", file=sys.stderr)
        return 1

    command = [
        "swift",
        "run",
        "--package-path",
        str(package_path),
        "ROISelector",
        *(sys.argv[1:] if argv is None else argv),
    ]
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as error:
        print(f"image-eval: error: missing executable: {error.filename}", file=sys.stderr)
        return 127
    return completed.returncode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
