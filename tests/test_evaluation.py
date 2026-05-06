from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.evaluation import evaluate_image, evaluation_result_to_dict


class EvaluationTests(unittest.TestCase):
    def test_evaluate_image_is_pure_and_returns_json_serializable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), (16, 4))

            result = evaluate_image(
                base_image=image,
                template=_template(width=16, height=16),
                subject_image=image.copy(),
            )

            report = evaluation_result_to_dict(result)
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(report["registration"]["mode"], "identity")
            self.assertNotIn("base_image_path", report["registered_template"])
            self.assertNotIn("path", report["registered_template"]["source_image"])
            self.assertGreaterEqual(len(report["mtf"]["rows"]), 1)
            self.assertGreaterEqual(len(report["nps"]["rows"]), 1)

    def test_rejects_template_shape_that_does_not_match_base_image(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "do not match base image"):
            evaluate_image(
                base_image=image,
                template=_template(width=7, height=4),
                subject_image=image,
            )


def _template(*, width: int, height: int) -> dict:
    return {
        "source_image": {"width": width, "height": height},
        "normalization_rois": {
            "black": {"x0": 0, "y0": 0, "x1": 2, "y1": height},
            "white": {"x0": 2, "y0": 0, "x1": 4, "y1": height},
        },
        "bar_rois": [
            {
                "group": 0,
                "element": 1,
                "orientation": "X",
                "rect": {"x0": 0, "y0": 0, "x1": width, "y1": height},
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
