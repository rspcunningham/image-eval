from __future__ import annotations

import unittest

import numpy as np

from image_eval.nps_results import (
    CYCLES_PER_PIXEL_FREQUENCY,
    SpatialFrequencyCalibration,
    calculate_nps_report,
    calculate_nps_results,
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

    def test_frequency_calibration_controls_reported_axis(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0
        calibration = SpatialFrequencyCalibration(
            unit="cycles/mm",
            cycles_per_pixel_multiplier=2.0,
        )

        default_report = calculate_nps_report(image, _template())
        calibrated_report = calculate_nps_report(
            image,
            _template(),
            frequency_calibration=calibration,
        )

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


def _template(*, width: int = 8, height: int = 4, split: int = 4) -> dict:
    return {
        "source_image": {"width": width, "height": height},
        "normalization_rois": {
            "black": {"x0": 0, "y0": 0, "x1": split, "y1": height},
            "white": {"x0": split, "y0": 0, "x1": width, "y1": height},
        },
        "bar_rois": [],
    }


if __name__ == "__main__":
    unittest.main()
