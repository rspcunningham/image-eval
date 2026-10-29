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
            self.assertNotIn("schema_version", report)
            self.assertNotIn("dqe", report)
            self.assertEqual(report["registration"]["mode"], "identity")
            self.assertNotIn("base_image_path", report["registered_template"])
            self.assertNotIn("path", report["registered_template"]["source_image"])
            self.assertGreaterEqual(len(report["mtf"]["rows"]), 1)
            self.assertNotIn("fit_profiles", report["mtf"])
            self.assertGreaterEqual(len(report["nps"]["rows"]), 1)
            self.assertEqual(
                set(report["mtf"]["rows"][0]),
                {"cycles_per_mm", "orientation", "mtf", "noise_mtf", "resolved"},
            )

    def test_evaluation_dict_includes_fit_details_when_requested(self) -> None:
        image = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), (16, 4))

        result = evaluate_image(
            base_image=image,
            template=_template(width=16, height=16),
            subject_image=image.copy(),
        )

        report = evaluation_result_to_dict(result, details=["fits"])
        fit_profiles = report["mtf"]["fit_profiles"]
        self.assertEqual(len(fit_profiles), 1)

        fit_profile = fit_profiles[0]
        self.assertEqual(
            {
                "group",
                "element",
                "orientation",
                "rect",
                "profile_axis",
                "frequency_lp_per_mm",
                "pixels_per_mm",
                "fit",
                "series",
                "mtf_points",
            },
            set(fit_profile),
        )
        # 4 cycles over 16 samples: the 3rd and 5th harmonics alias, so only the fundamental is fit.
        self.assertEqual(fit_profile["fit"]["harmonics"], [1])
        self.assertEqual(fit_profile["series"]["sample_index"], list(range(16)))
        self.assertEqual(len(fit_profile["series"]["normalized_position"]), 16)
        self.assertEqual(len(fit_profile["series"]["observed_profile"]), 16)
        self.assertEqual(len(fit_profile["series"]["fitted_profile"]), 16)
        self.assertEqual(len(fit_profile["series"]["residuals"]), 16)
        self.assertEqual([point["harmonic"] for point in fit_profile["mtf_points"]], [1])

    def test_evaluation_dict_expands_all_details_to_fit_details(self) -> None:
        image = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), (16, 4))

        result = evaluate_image(
            base_image=image,
            template=_template(width=16, height=16),
            subject_image=image.copy(),
        )

        report = evaluation_result_to_dict(result, details=["all"])

        self.assertIn("fit_profiles", report["mtf"])

    def test_rejects_template_shape_that_does_not_match_base_image(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "do not match base image"):
            evaluate_image(
                base_image=image,
                template=_template(width=7, height=4),
                subject_image=image,
            )

    def test_rejects_normalization_roi_outside_registered_subject_image(self) -> None:
        image = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), (16, 4))
        template = _template(width=16, height=16)
        template["normalization_rois"]["black"] = {"x0": 15, "y0": 0, "x1": 17, "y1": 16}

        with self.assertRaisesRegex(ValueError, "normalization ROI must be fully visible"):
            evaluate_image(
                base_image=image,
                template=template,
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
