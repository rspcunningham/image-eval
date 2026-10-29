from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from image_eval.cli import main


class ImageEvalCLITests(unittest.TestCase):
    def test_eval_writes_json_stdout_for_explicit_base_template_and_subject_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([
                    "eval",
                    "--base-url",
                    str(image_path),
                    "--template",
                    str(template_path),
                    "--subject-url",
                    str(image_path),
                ])

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            self.assertNotIn("schema_version", report)
            self.assertNotIn("dqe", report)
            self.assertEqual(report["registration"]["mode"], "identity")
            self.assertEqual(report["image_shapes"]["base"], {"height": 16, "width": 16})
            self.assertNotIn("base_image_path", report["registered_template"])
            self.assertNotIn("path", report["registered_template"]["source_image"])
            self.assertEqual(report["mtf"]["frequency_unit"], "cycles/mm")
            self.assertEqual(report["nps"]["frequency_unit"], "cycles/mm")
            self.assertNotIn("fit_profiles", report["mtf"])
            self.assertGreaterEqual(len(report["mtf"]["rows"]), 1)
            self.assertGreaterEqual(len(report["nps"]["rows"]), 1)
            self.assertEqual(
                set(report["mtf"]["rows"][0]),
                {"cycles_per_mm", "orientation", "mtf", "noise_mtf", "resolved"},
            )

    def test_detail_fits_adds_plot_ready_square_wave_fit_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([
                    "eval",
                    "--base-url",
                    image_path.as_uri(),
                    "--template",
                    template_path.as_uri(),
                    "--subject-url",
                    image_path.as_uri(),
                    "--detail",
                    "fits",
                    "--detail",
                    "all",
                ])

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            fit_profiles = report["mtf"]["fit_profiles"]
            self.assertEqual(len(fit_profiles), 1)

            fit_profile = fit_profiles[0]
            self.assertEqual(fit_profile["group"], 0)
            self.assertEqual(fit_profile["element"], 1)
            self.assertEqual(fit_profile["orientation"], "X")
            self.assertEqual(fit_profile["profile_axis"], "x")
            self.assertEqual(fit_profile["frequency_lp_per_mm"], 1.0)
            self.assertEqual(fit_profile["rect"], {"x0": 0, "y0": 0, "x1": 16, "y1": 16})
            self.assertGreater(fit_profile["pixels_per_mm"], 0)

            fit = fit_profile["fit"]
            # 4 cycles over 16 samples: the 3rd and 5th harmonics alias, so only the fundamental is fit.
            self.assertEqual(fit["terms"], 1)
            self.assertEqual(fit["harmonics"], [1])
            self.assertIn("cycles", fit)
            self.assertIn("offset", fit)
            self.assertIn("baseline_slope", fit)
            self.assertIn("sine_coefficients", fit)
            self.assertIn("cosine_coefficients", fit)
            self.assertIn("fundamental_amplitude", fit)
            self.assertIn("residual_rms", fit)

            series = fit_profile["series"]
            self.assertEqual(series["sample_index"], list(range(16)))
            self.assertEqual(len(series["normalized_position"]), 16)
            self.assertEqual(len(series["observed_profile"]), 16)
            self.assertEqual(len(series["fitted_profile"]), 16)
            self.assertEqual(len(series["residuals"]), 16)

            self.assertEqual(
                [point["harmonic"] for point in fit_profile["mtf_points"]],
                [1],
            )
            self.assertEqual(
                [point["cycles_per_mm"] for point in fit_profile["mtf_points"]],
                [1.0],
            )

    def test_eval_accepts_percentile_norm_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([
                    "eval",
                    "--base-url",
                    str(image_path),
                    "--template",
                    str(template_path),
                    "--subject-url",
                    str(image_path),
                    "--norm",
                    "0.05",
                ])

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            self.assertGreaterEqual(len(report["mtf"]["rows"]), 1)

    def test_invalid_norm_cutoff_is_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as error:
            main([
                "eval",
                "--base-url",
                "base.npy",
                "--template",
                "template.json",
                "--subject-url",
                "subject.npy",
                "--norm",
                "0.5",
            ])

        self.assertEqual(error.exception.code, 2)

    def test_removed_plot_flags_are_usage_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)

            for removed_flag in ("--plots", "--no-plots"):
                args = [
                    "eval",
                    "--base-url",
                    str(image_path),
                    "--template",
                    str(template_path),
                    "--subject-url",
                    str(image_path),
                    removed_flag,
                ]
                if removed_flag == "--plots":
                    args.append("mtf")

                with self.subTest(flag=removed_flag):
                    with self.assertRaises(SystemExit) as error:
                        main(args)
                    self.assertEqual(error.exception.code, 2)

    def test_removed_output_flags_are_usage_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)

            cases = [
                ["--json"],
                ["--out", str(root / "outputs")],
            ]
            for removed_args in cases:
                args = [
                    "eval",
                    "--base-url",
                    str(image_path),
                    "--template",
                    str(template_path),
                    "--subject-url",
                    str(image_path),
                    *removed_args,
                ]
                with self.subTest(args=removed_args):
                    with self.assertRaises(SystemExit) as error:
                        main(args)
                    self.assertEqual(error.exception.code, 2)

    def test_missing_subcommand_exits_with_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as error:
            main([])

        self.assertEqual(error.exception.code, 2)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    image_path = root / "base.npy"
    template_path = root / "template.json"

    image = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), (16, 4))
    np.save(image_path, image)
    template = {
        "source_image": {
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
    template_path.write_text(json.dumps(template))
    return image_path, template_path


if __name__ == "__main__":
    unittest.main()
