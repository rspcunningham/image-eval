from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.nps_results import (
    CYCLES_PER_PIXEL_FREQUENCY,
    SpatialFrequencyCalibration,
    calculate_nps_report,
    calculate_nps_results,
    main,
    nps_csv_columns,
    save_nps_report,
    save_nps_results_csv,
)


class NPSResultsTests(unittest.TestCase):
    def test_constant_normalization_rois_have_zero_nps(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0
        template = _template()

        report = calculate_nps_report(image, template)

        self.assertEqual(report.frequency_calibration, CYCLES_PER_PIXEL_FREQUENCY)
        self.assertEqual(len(report.spectra), 2)
        self.assertEqual(len(report.results), 2)
        for result in report.results:
            self.assertEqual(result.black_nps, 0.0)
            self.assertEqual(result.white_nps, 0.0)
            self.assertEqual(result.average_nps, 0.0)

    def test_deterministic_noise_produces_nonzero_nps(self) -> None:
        stripe = np.tile(np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64), (8, 2))
        image = np.zeros((8, 16), dtype=np.float64)
        image[:, 0:8] = 10.0 + stripe
        image[:, 8:16] = 20.0 + stripe

        results = calculate_nps_results(image, _template(width=16, height=8, split=8))

        self.assertTrue(any((result.black_nps or 0.0) > 0.0 for result in results))
        self.assertTrue(any((result.white_nps or 0.0) > 0.0 for result in results))
        self.assertTrue(any((result.average_nps or 0.0) > 0.0 for result in results))

    def test_frequency_calibration_controls_reported_axis_and_csv_header(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0
        calibration = SpatialFrequencyCalibration(
            unit="lp/mm",
            cycles_per_pixel_multiplier=2.0,
        )

        default_report = calculate_nps_report(image, _template())
        calibrated_report = calculate_nps_report(
            image,
            _template(),
            frequency_calibration=calibration,
        )

        self.assertEqual(nps_csv_columns(calibration)[0], "LP per MM")
        self.assertAlmostEqual(
            calibrated_report.results[0].frequency,
            default_report.results[0].frequency * 2.0,
        )

    def test_rejects_non_finite_nps_roi_pixels(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0
        image[0, 0] = np.nan

        with self.assertRaises(ValueError):
            calculate_nps_report(image, _template())

    def test_saves_report_directory_with_summary_and_spectrum_plots(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0
        report = calculate_nps_report(image, _template())

        with tempfile.TemporaryDirectory() as directory:
            paths = save_nps_report(report, Path(directory))

            self.assertTrue(paths.csv_path.exists())
            self.assertTrue(paths.plot_path.exists())
            self.assertEqual(len(paths.spectrum_paths), 2)
            self.assertEqual(paths.spectrum_paths[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(paths.spectrum_paths[1].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_saves_requested_csv_columns(self) -> None:
        results = [
            calculate_nps_report(np.hstack([np.zeros((4, 4)), np.ones((4, 4))]), _template()).results[
                0
            ]
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nps.csv"

            save_nps_results_csv(results, output_path)

            with output_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(list(rows[0].keys()), nps_csv_columns(CYCLES_PER_PIXEL_FREQUENCY))
            self.assertEqual(rows[0]["black NPS"], "0")
            self.assertEqual(rows[0]["white NPS"], "0")
            self.assertEqual(rows[0]["average NPS"], "0")

    def test_cli_loads_image_path_from_template_base_image_path(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            template_path = root / "template.json"
            output_dir = root / "report"
            np.save(image_path, image)
            template_path.write_text(json.dumps({
                **_template(),
                "base_image_path": "image.npy",
                "source_image": {"path": "image.npy", "width": 8, "height": 4},
            }))

            exit_code = main([str(template_path), str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "nps.csv").exists())
            self.assertTrue((output_dir / "nps.png").exists())
            self.assertTrue((output_dir / "nps_spectra" / "black_2d.png").exists())
            self.assertTrue((output_dir / "nps_spectra" / "white_2d.png").exists())


def _template(*, width: int = 8, height: int = 4, split: int = 4) -> dict:
    return {
        "base_image_path": "image.npy",
        "source_image": {"path": "image.npy", "width": width, "height": height},
        "normalization_rois": {
            "black": {"x0": 0, "y0": 0, "x1": split, "y1": height},
            "white": {"x0": split, "y0": 0, "x1": width, "y1": height},
        },
        "bar_rois": [],
    }


if __name__ == "__main__":
    unittest.main()
