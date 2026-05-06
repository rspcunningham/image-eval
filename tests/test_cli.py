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
    def test_json_stdout_uses_explicit_base_template_and_subject_sources(self) -> None:
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
                    "--json",
                ])

            self.assertEqual(exit_code, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["registration"]["mode"], "identity")
            self.assertEqual(report["image_shapes"]["base"], {"height": 16, "width": 16})
            self.assertNotIn("base_image_path", report["registered_template"])
            self.assertNotIn("path", report["registered_template"]["source_image"])
            self.assertEqual(report["mtf"]["frequency_unit"], "lp/mm")
            self.assertEqual(report["nps"]["frequency_unit"], "lp/mm")
            self.assertEqual(report["dqe"]["frequency_unit"], "lp/mm")
            self.assertGreaterEqual(len(report["mtf"]["rows"]), 1)
            self.assertGreaterEqual(len(report["nps"]["rows"]), 1)

    def test_out_writes_report_and_default_plot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)
            output_dir = root / "outputs"

            exit_code = main([
                "eval",
                "--base-url",
                image_path.as_uri(),
                "--template",
                template_path.as_uri(),
                "--subject-url",
                image_path.as_uri(),
                "--out",
                str(output_dir),
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "report.json").exists())
            self.assertTrue((output_dir / "mtf.png").exists())
            self.assertTrue((output_dir / "nps.png").exists())
            self.assertTrue((output_dir / "dqe.png").exists())
            self.assertTrue((output_dir / "roi_fits" / "001_g0_e1_x_fit.png").exists())
            self.assertTrue((output_dir / "nps_spectra" / "black_2d.png").exists())
            self.assertTrue((output_dir / "registration" / "registration.json").exists())

    def test_no_plots_writes_only_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)
            output_dir = root / "outputs"

            exit_code = main([
                "eval",
                "--base-url",
                str(image_path),
                "--template",
                str(template_path),
                "--subject-url",
                str(image_path),
                "--out",
                str(output_dir),
                "--no-plots",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "report.json").exists())
            self.assertFalse((output_dir / "mtf.png").exists())
            self.assertFalse((output_dir / "registration").exists())

    def test_plot_selection_writes_only_requested_plot_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, template_path = _write_inputs(root)
            output_dir = root / "outputs"

            exit_code = main([
                "eval",
                "--base-url",
                str(image_path),
                "--template",
                str(template_path),
                "--subject-url",
                str(image_path),
                "--out",
                str(output_dir),
                "--plots",
                "mtf,nps",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "mtf.png").exists())
            self.assertTrue((output_dir / "nps.png").exists())
            self.assertFalse((output_dir / "dqe.png").exists())
            self.assertFalse((output_dir / "roi_fits").exists())

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
