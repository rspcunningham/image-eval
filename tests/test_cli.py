from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.cli import evaluate_image


class ImageEvalCLITests(unittest.TestCase):
    def test_evaluates_base_image_with_identity_registration_and_fixed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "base.npy"
            template_path = root / "template.json"
            output_dir = root / "outputs"

            image = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), (16, 4))
            np.save(image_path, image)
            template = {
                "base_image_path": str(image_path),
                "source_image": {
                    "path": str(image_path),
                    "width": 16,
                    "height": 16,
                },
                "normalization_rois": {
                    "black": {"x0": 0, "y0": 0, "x1": 2, "y1": 16},
                    "white": {"x0": 2, "y0": 0, "x1": 4, "y1": 16},
                },
                "bar_rois": [
                    {
                        "group": 0,
                        "element": 1,
                        "orientation": "X",
                        "rect": {"x0": 0, "y0": 0, "x1": 16, "y1": 16},
                    }
                ],
            }
            with template_path.open("w") as file:
                json.dump(template, file)

            paths = evaluate_image(image_path, template_path, output_dir)

            self.assertTrue(paths.mtf_paths.csv_path.exists())
            with paths.mtf_paths.csv_path.open(newline="") as file:
                self.assertEqual(next(csv.reader(file)), ["cycles/mm", "orientation", "mtf"])
            self.assertTrue(paths.nps_paths.csv_path.exists())
            with paths.nps_paths.csv_path.open(newline="") as file:
                self.assertEqual(next(csv.reader(file))[0], "LP per MM")

            registration_paths = paths.registration_paths
            self.assertTrue(registration_paths.registration_json_path.exists())
            self.assertTrue(registration_paths.registered_template_path.exists())

            with registration_paths.registration_json_path.open() as file:
                registration = json.load(file)
            self.assertEqual(registration["mode"], "identity")
            self.assertEqual(
                registration["transform_subject_to_base"],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )

            with registration_paths.registered_template_path.open() as file:
                registered_template = json.load(file)
            self.assertEqual(registered_template["source_image"]["width"], 16)
            self.assertEqual(registered_template["source_image"]["height"], 16)
            self.assertEqual(len(registered_template["bar_rois"]), 1)


if __name__ == "__main__":
    unittest.main()
