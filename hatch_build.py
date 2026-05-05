from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if os.environ.get("IMAGE_EVAL_SKIP_SWIFT_BUILD"):
            return

        package_path = Path(self.root) / "native" / "ROISelector"
        if not package_path.exists():
            raise RuntimeError(f"Swift package not found: {package_path}")

        configuration = os.environ.get("IMAGE_EVAL_SWIFT_CONFIGURATION", "debug")
        subprocess.run(
            [
                "swift",
                "build",
                "--package-path",
                str(package_path),
                "--product",
                "ROISelector",
                "-c",
                configuration,
            ],
            check=True,
        )
