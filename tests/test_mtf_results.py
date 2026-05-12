from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.mtf_profiles import BarROIProfile
from image_eval.mtf_results import (
    MTF_CSV_COLUMNS,
    average_pixels_per_mm_from_fits,
    calculate_mtf_results,
    calculate_mtf_report,
    line_profile_mtf_points,
    main,
    mtf_results_from_fits,
    roi_pixels_per_mm,
    save_mtf_report,
    save_mtf_results_csv,
)
from image_eval.square_wave_fit import _square_wave_design_matrix, fit_bar_roi_profiles


class MTFResultsTests(unittest.TestCase):
    def test_line_profile_mtf_points_returns_fitted_odd_harmonics(self) -> None:
        coefficients = np.array([0.5, 0.0, 0.2, 0.0, 0.1, 0.0, 0.04, 0.0])
        profile = _square_wave_design_matrix(64, cycles=3.21) @ coefficients

        points = line_profile_mtf_points(profile, 16.0)

        self.assertEqual(list(points.keys()), [16.0, 48.0, 80.0])
        self.assertAlmostEqual(points[16.0], 0.2 * math.pi / 2.0)
        self.assertAlmostEqual(points[48.0], 0.1 * math.pi * 3 / 2.0)
        self.assertAlmostEqual(points[80.0], 0.04 * math.pi * 5 / 2.0)

    def test_mtf_results_preserve_duplicate_frequency_orientation_rows(self) -> None:
        fitted_profiles = [
            *_fits("X", fundamental_amplitude=0.2, third_amplitude=0.1, fifth_amplitude=0.04),
            *_fits("X", fundamental_amplitude=0.3, third_amplitude=0.08, fifth_amplitude=0.02),
        ]

        results = mtf_results_from_fits(fitted_profiles)

        self.assertEqual(len(results), 6)
        self.assertEqual([result.cycles_per_mm for result in results], [16, 48, 80, 16, 48, 80])
        self.assertEqual([result.orientation for result in results], ["X"] * 6)
        self.assertAlmostEqual(results[0].mtf, 0.2 * math.pi / 2.0)
        self.assertAlmostEqual(results[1].mtf, 0.1 * math.pi * 3 / 2.0)
        self.assertAlmostEqual(results[3].mtf, 0.3 * math.pi / 2.0)

    def test_calculates_pixels_per_millimetre_from_fitted_cycles(self) -> None:
        fitted_profile = fit_bar_roi_profiles([_roi_profile("X", fundamental_amplitude=0.2)])[0]

        pixels_per_mm = roi_pixels_per_mm(fitted_profile)

        self.assertAlmostEqual(pixels_per_mm, 16 * 64 / fitted_profile.fit.cycles)

    def test_averages_pixels_per_millimetre_across_fitted_rois(self) -> None:
        fitted_profiles = fit_bar_roi_profiles([
            _roi_profile("X", fundamental_amplitude=0.2),
            _roi_profile("Y", fundamental_amplitude=0.4),
        ])

        pixels_per_mm = average_pixels_per_mm_from_fits(fitted_profiles)

        expected = np.mean([
            roi_pixels_per_mm(fitted_profile) for fitted_profile in fitted_profiles
        ])
        self.assertAlmostEqual(pixels_per_mm, expected)

    def test_rejects_average_pixels_per_millimetre_without_fits(self) -> None:
        with self.assertRaises(ValueError):
            average_pixels_per_mm_from_fits([])

    def test_saves_requested_csv_columns_and_duplicate_rows(self) -> None:
        results = mtf_results_from_fits([
            *_fits("X", fundamental_amplitude=0.2),
            *_fits("X", fundamental_amplitude=0.4),
        ])

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mtf.csv"

            save_mtf_results_csv(results, output_path)

            with output_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(list(rows[0].keys()), MTF_CSV_COLUMNS)
            self.assertEqual(rows[0]["cycles/mm"], "16")
            self.assertEqual(rows[0]["orientation"], "X")
            self.assertNotEqual(rows[0]["mtf"], "")
            self.assertEqual([row["cycles/mm"] for row in rows].count("16"), 2)

    def test_calculates_results_from_image_and_template(self) -> None:
        image = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.float64,
        )
        template = {
            "normalization_rois": {
                "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 4},
                "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 4},
            },
            "bar_rois": [
                {
                    "group": 0,
                    "element": 1,
                    "orientation": "X",
                    "rect": {"x0": 0, "y0": 0, "x1": 8, "y1": 4},
                }
            ],
        }

        results = calculate_mtf_results(image, template)

        self.assertEqual(len(results), 3)
        self.assertEqual([result.cycles_per_mm for result in results], [1, 3, 5])
        self.assertEqual([result.orientation for result in results], ["X", "X", "X"])

    def test_saves_report_directory_with_summary_csv_only(self) -> None:
        image = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.float64,
        )
        template = {
            "normalization_rois": {
                "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 4},
                "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 4},
            },
            "bar_rois": [
                {
                    "group": 0,
                    "element": 1,
                    "orientation": "X",
                    "rect": {"x0": 0, "y0": 0, "x1": 8, "y1": 4},
                }
            ],
        }
        report = calculate_mtf_report(image, template)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = save_mtf_report(report, root)

            self.assertTrue(paths.csv_path.exists())

    def test_cli_loads_image_path_from_template_base_image_path(self) -> None:
        image = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            template_path = root / "template.json"
            output_dir = root / "report"
            np.save(image_path, image)
            template_path.write_text(json.dumps({
                "base_image_path": "image.npy",
                "source_image": {"path": "image.npy", "width": 8, "height": 4},
                "normalization_rois": {
                    "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 4},
                    "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 4},
                },
                "bar_rois": [
                    {
                        "group": 0,
                        "element": 1,
                        "orientation": "X",
                        "rect": {"x0": 0, "y0": 0, "x1": 8, "y1": 4},
                    }
                ],
            }))

            exit_code = main([str(template_path), str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "mtf.csv").exists())


def _fits(
    orientation: str,
    *,
    fundamental_amplitude: float,
    third_amplitude: float = 0.0,
    fifth_amplitude: float = 0.0,
):
    return fit_bar_roi_profiles([
        _roi_profile(
            orientation,
            fundamental_amplitude=fundamental_amplitude,
            third_amplitude=third_amplitude,
            fifth_amplitude=fifth_amplitude,
        )
    ])


def _roi_profile(
    orientation: str,
    *,
    fundamental_amplitude: float,
    third_amplitude: float = 0.0,
    fifth_amplitude: float = 0.0,
) -> BarROIProfile:
    profile = _square_wave_design_matrix(64, cycles=3.0) @ np.array(
        [
            0.5,
            0.0,
            fundamental_amplitude,
            0.0,
            third_amplitude,
            0.0,
            fifth_amplitude,
            0.0,
        ]
    )
    return BarROIProfile(
        group=4,
        element=1,
        orientation=orientation,
        frequency_lp_per_mm=16,
        rect={"x0": 0, "y0": 0, "x1": 64, "y1": 8},
        profile_axis="x" if orientation == "X" else "y",
        profile=profile,
    )


if __name__ == "__main__":
    unittest.main()
