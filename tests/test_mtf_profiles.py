from __future__ import annotations

import unittest

import numpy as np

from image_eval.mtf_profiles import (
    bar_roi_profiles,
    oriented_samples,
    normalize_image_intensity,
    prepare_mtf_profiles,
)


class MTFProfileTests(unittest.TestCase):
    def test_normalizes_image_using_black_and_white_roi_means(self) -> None:
        image = np.array(
            [
                [10, 10, 20, 30, 30],
                [20, 30, 40, 50, 60],
            ],
            dtype=np.float32,
        )
        normalization_rois = {
            "black": {"x0": 0, "y0": 0, "x1": 2, "y1": 1},
            "white": {"x0": 3, "y0": 0, "x1": 5, "y1": 1},
        }

        normalized = normalize_image_intensity(image, normalization_rois)

        self.assertEqual(normalized.normalization.black_mean, 10)
        self.assertEqual(normalized.normalization.white_mean, 30)
        self.assertEqual(normalized.normalization_rois.black, normalization_rois["black"])
        self.assertEqual(normalized.normalization_rois.white, normalization_rois["white"])
        np.testing.assert_allclose(
            normalized.image,
            np.array(
                [
                    [0, 0, 0.5, 1, 1],
                    [0.5, 1, 1.5, 2, 2.5],
                ],
                dtype=np.float64,
            ),
        )

    def test_normalizes_image_using_quantile_cutoffs(self) -> None:
        image = np.array([[10, 40, 60, 90]], dtype=np.float32)
        normalization_rois = {
            "black": {"x0": 0, "y0": 0, "x1": 2, "y1": 1},
            "white": {"x0": 2, "y0": 0, "x1": 4, "y1": 1},
        }

        normalized = normalize_image_intensity(image, normalization_rois, norm=0.0)

        self.assertEqual(normalized.normalization.method, "quantile")
        self.assertEqual(normalized.normalization.cutoff, 0.0)
        self.assertEqual(normalized.normalization.black_level, 10)
        self.assertEqual(normalized.normalization.white_level, 90)
        np.testing.assert_allclose(
            normalized.image,
            np.array([[0, 0.375, 0.625, 1]], dtype=np.float64),
        )

    def test_accepts_inverted_normalization_scale(self) -> None:
        image = np.array([[30, 10]], dtype=np.float32)
        normalization_rois = {
            "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
        }

        normalized = normalize_image_intensity(image, normalization_rois)

        self.assertEqual(normalized.normalization.black_mean, 30)
        self.assertEqual(normalized.normalization.white_mean, 10)
        np.testing.assert_allclose(normalized.image, np.array([[0, 1]], dtype=np.float64))

    def test_rejects_zero_normalization_scale(self) -> None:
        image = np.ones((3, 3), dtype=np.float32)
        normalization_rois = {
            "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
        }

        with self.assertRaises(ValueError):
            normalize_image_intensity(image, normalization_rois)

    def test_rejects_non_object_normalization_rois(self) -> None:
        image = np.ones((3, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "normalization_rois must be a JSON object"):
            normalize_image_intensity(image, None)

    def test_rejects_invalid_numeric_normalization_cutoff(self) -> None:
        image = np.ones((3, 3), dtype=np.float32)
        normalization_rois = {
            "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
        }

        with self.assertRaisesRegex(ValueError, "numeric normalization cutoff"):
            normalize_image_intensity(image, normalization_rois, norm=0.5)

    def test_samples_x_roi_across_columns_and_trims_bar_ends(self) -> None:
        image = np.arange(100, dtype=np.float64).reshape(10, 10)
        bar_rois = [
            {
                "group": 4,
                "element": 1,
                "orientation": "X",
                "rect": {"x0": 2, "y0": 0, "x1": 7, "y1": 10},
            }
        ]

        (profile,) = bar_roi_profiles(image, bar_rois, 0.0)

        self.assertEqual(profile.frequency_lp_per_mm, 16)
        self.assertEqual(profile.samples.width_px, 5)
        rows = profile.samples.values // 10
        self.assertEqual(set(rows), {2, 3, 4, 5, 6, 7})  # middle 60% of 10 rows
        np.testing.assert_allclose(np.unique(profile.samples.positions), [0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(profile.samples.values % 10, profile.samples.positions * 5 + 1.5)

    def test_samples_y_roi_across_rows(self) -> None:
        image = np.arange(100, dtype=np.float64).reshape(10, 10)
        rect = {"x0": 0, "y0": 3, "x1": 10, "y1": 7}

        samples = oriented_samples(image, rect, "Y", 0.0)

        self.assertEqual(samples.width_px, 4)
        np.testing.assert_allclose(samples.values // 10, samples.positions * 4 + 2.5)

    def test_rotated_positions_follow_the_bars(self) -> None:
        # Image value = signed distance across bars turned by 2 degrees about the rect centre.
        angle = np.radians(2.0)
        y, x = np.mgrid[:60, :60] + 0.5
        image = (x - 30) * np.cos(angle) + (y - 30) * np.sin(angle)

        samples = oriented_samples(image, {"x0": 0, "y0": 0, "x1": 60, "y1": 60}, "X", 2.0)

        np.testing.assert_allclose(samples.values, (samples.positions - 0.5) * samples.width_px)
        self.assertAlmostEqual(samples.width_px, 60 / (np.cos(angle) + np.sin(angle)), places=9)

    def test_prepare_mtf_profiles_runs_normalization_then_bar_profiles(self) -> None:
        image = np.array(
            [
                [10, 30, 10, 30],
                [10, 30, 20, 40],
                [10, 30, 30, 50],
            ],
            dtype=np.float32,
        )
        template = {
            "normalization_rois": {
                "black": {"x0": 0, "y0": 0, "x1": 1, "y1": 3},
                "white": {"x0": 1, "y0": 0, "x1": 2, "y1": 3},
            },
            "bar_rois": [
                {
                    "group": 0,
                    "element": 1,
                    "orientation": "X",
                    "rect": {"x0": 2, "y0": 0, "x1": 4, "y1": 3},
                }
            ],
        }

        prepared = prepare_mtf_profiles(image, template)

        self.assertEqual(prepared.normalization.black_mean, 10)
        self.assertEqual(prepared.normalization.white_mean, 30)
        self.assertEqual(len(prepared.bar_profiles), 1)
        np.testing.assert_allclose(np.sort(prepared.bar_profiles[0].samples.values), [0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
