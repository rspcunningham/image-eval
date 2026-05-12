from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_eval.initialize import main


class ImageEvalInitializeTests(unittest.TestCase):
    def test_runs_embedded_roi_selector_with_template_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary_path = Path(directory) / "ROISelector"
            binary_path.write_text("")
            completed = subprocess.CompletedProcess(args=[], returncode=0)

            with (
                patch("image_eval.initialize._roi_selector_binary", return_value=binary_path),
                patch("image_eval.initialize.subprocess.run", return_value=completed) as run,
            ):
                exit_code = main([
                    "source.npy",
                    "template.json",
                    "--groups",
                    "4-7",
                    "--elements",
                    "1-6",
                ])

            self.assertEqual(exit_code, 0)
            run.assert_called_once_with(
                [
                    str(binary_path),
                    "source.npy",
                    "template.json",
                    "--groups",
                    "4-7",
                    "--elements",
                    "1-6",
                ],
                check=False,
            )


if __name__ == "__main__":
    unittest.main()
