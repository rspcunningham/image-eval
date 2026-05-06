from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SwiftROISelectorBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return

        _validate_platform()
        binary_path = _build_roi_selector(Path(self.root))

        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-macosx_13_0_arm64"
        build_data["force_include"][str(binary_path)] = "image_eval/_bin/ROISelector"


def _validate_platform() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("ROISelector wheels can only be built on Apple Silicon macOS")

    if shutil.which("swift") is None:
        raise RuntimeError(
            "Swift is required to build the embedded ROISelector executable. "
            "Install Xcode command line tools and retry."
        )


def _build_roi_selector(root: Path) -> Path:
    package_path = root / "native" / "ROISelector"
    if not package_path.exists():
        raise RuntimeError(f"Swift package not found: {package_path}")

    command = [
        "swift",
        "build",
        "-c",
        "release",
        "--package-path",
        str(package_path),
        "--product",
        "ROISelector",
        "--arch",
        "arm64",
    ]
    subprocess.run(command, check=True)

    bin_path_command = [*command, "--show-bin-path"]
    completed = subprocess.run(
        bin_path_command,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    binary_path = Path(completed.stdout.strip()) / "ROISelector"
    if not binary_path.exists():
        raise RuntimeError(f"Swift build did not produce ROISelector at {binary_path}")

    binary_path.chmod(os.stat(binary_path).st_mode | 0o755)
    return binary_path
