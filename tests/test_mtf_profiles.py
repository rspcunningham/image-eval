from __future__ import annotations

import unittest

import numpy as np

from image_eval.mtf_profiles import (
    bar_roi_profiles,
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

    def test_collapses_x_oriented_roi_over_y_axis(self) -> None:
        image = np.arange(20, dtype=np.float64).reshape(4, 5)
        bar_rois = [
            {
                "group": 4,
                "element": 1,
                "orientation": "X",
                "rect": {"x0": 1, "y0": 1, "x1": 4, "y1": 3},
            }
        ]

        profiles = bar_roi_profiles(image, bar_rois)

        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].profile_axis, "x")
        self.assertEqual(profiles[0].frequency_lp_per_mm, 16)
        np.testing.assert_allclose(profiles[0].profile, np.array([8.5, 9.5, 10.5]))

    def test_collapses_y_oriented_roi_over_x_axis(self) -> None:
        image = np.arange(20, dtype=np.float64).reshape(4, 5)
        bar_rois = [
            {
                "group": 4,
                "element": 2,
                "orientation": "Y",
                "rect": {"x0": 1, "y0": 1, "x1": 4, "y1": 3},
            }
        ]

        profiles = bar_roi_profiles(image, bar_rois)

        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].profile_axis, "y")
        np.testing.assert_allclose(profiles[0].profile, np.array([7, 12]))

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
        np.testing.assert_allclose(prepared.bar_profiles[0].profile, np.array([0.5, 1.5]))


if __name__ == "__main__":
    unittest.main()
