from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.mtf_profiles import BarROIProfile
from image_eval.mtf_results import (
    FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION,
    MTF_CSV_COLUMNS,
    average_pixels_per_mm_from_fits,
    calculate_mtf_results,
    calculate_mtf_report,
    main,
    mtf_results_from_fits,
    roi_pixels_per_mm,
    save_mtf_report,
    save_mtf_results_csv,
)
from image_eval.square_wave_fit import fit_bar_roi_profiles, square_wave_design_matrix


class MTFResultsTests(unittest.TestCase):
    def test_groups_x_and_y_mtf_by_frequency_and_averages_both_orientations(self) -> None:
        fitted_profiles = fit_bar_roi_profiles([
            _roi_profile("X", fundamental_amplitude=0.2),
            _roi_profile("Y", fundamental_amplitude=0.4),
        ])

        results = mtf_results_from_fits(fitted_profiles)

        self.assertEqual(len(results), 1)
        expected_x = 0.2 * FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION
        expected_y = 0.4 * FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION
        self.assertAlmostEqual(results[0].frequency_lp_per_mm, 16)
        self.assertAlmostEqual(results[0].x_mtf or 0, expected_x)
        self.assertAlmostEqual(results[0].y_mtf or 0, expected_y)
        self.assertAlmostEqual(results[0].average_mtf, (expected_x + expected_y) / 2)

    def test_average_uses_available_orientation_when_other_orientation_is_missing(self) -> None:
        fitted_profiles = fit_bar_roi_profiles([_roi_profile("X", fundamental_amplitude=0.2)])

        results = mtf_results_from_fits(fitted_profiles)

        self.assertAlmostEqual(
            results[0].average_mtf,
            0.2 * FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION,
        )
        self.assertIsNone(results[0].y_mtf)

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

    def test_saves_requested_csv_columns(self) -> None:
        results = mtf_results_from_fits([*_fits("X", 0.2), *_fits("Y", 0.4)])

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mtf.csv"

            save_mtf_results_csv(results, output_path)

            with output_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(list(rows[0].keys()), MTF_CSV_COLUMNS)
            self.assertEqual(rows[0]["LP per MM"], "16")
            self.assertNotEqual(rows[0]["XMTF"], "")
            self.assertNotEqual(rows[0]["YMTF"], "")
            self.assertNotEqual(rows[0]["average MTF"], "")

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

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].frequency_lp_per_mm, 1)
        self.assertIsNotNone(results[0].x_mtf)
        self.assertIsNone(results[0].y_mtf)

    def test_saves_report_directory_with_summary_and_roi_fit_plots(self) -> None:
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
            paths = save_mtf_report(report, Path(directory))

            self.assertTrue(paths.csv_path.exists())
            self.assertTrue(paths.plot_path.exists())
            self.assertEqual(len(paths.roi_fit_paths), 1)
            self.assertTrue(paths.roi_fit_paths[0].exists())
            self.assertEqual(paths.roi_fit_paths[0].name, "001_g0_e1_x_fit.png")

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
            self.assertTrue((output_dir / "mtf.png").exists())
            self.assertTrue((output_dir / "roi_fits" / "001_g0_e1_x_fit.png").exists())


def _fits(orientation: str, fundamental_amplitude: float):
    return fit_bar_roi_profiles([_roi_profile(orientation, fundamental_amplitude=fundamental_amplitude)])


def _roi_profile(orientation: str, *, fundamental_amplitude: float) -> BarROIProfile:
    profile = square_wave_design_matrix(64) @ np.array(
        [0.5, 0.0, fundamental_amplitude, 0.0, 0.0, 0.0, 0.0, 0.0]
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
