from __future__ import annotations

import unittest

import cv2
import numpy as np

from image_eval.mtf_results import calculate_mtf_report
from image_eval.usaf_1951 import line_pairs_per_mm

PIXELS_PER_MM = 1092.0
SIGMA_PX = 0.8  # Gaussian PSF
OVERSAMPLE = 8
BLACK, WHITE = 0.05, 0.95
MARGIN_PX = 12
GROUPS = range(4, 9)


def true_mtf(cycles_per_mm: float, angle_degrees: float = 0.0) -> float:
    """Gaussian PSF times the pixel's own box aperture, along bars rotated by the angle."""
    f = cycles_per_mm / PIXELS_PER_MM
    angle = np.radians(angle_degrees)
    return float(
        np.exp(-2 * np.pi**2 * SIGMA_PX**2 * f**2)
        * np.sinc(f * np.cos(angle))
        * np.sinc(f * np.sin(angle))
    )


def render(dark: np.ndarray) -> np.ndarray:
    """Gaussian-blur an oversampled indicator, then average each pixel's subsamples."""
    sigma = SIGMA_PX * OVERSAMPLE
    blurred = cv2.GaussianBlur(dark, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    height, width = dark.shape[0] // OVERSAMPLE, dark.shape[1] // OVERSAMPLE
    return blurred.reshape(height, OVERSAMPLE, width, OVERSAMPLE).mean(axis=(1, 3))


def subsample_grid(size_px: int) -> np.ndarray:
    return (np.arange(size_px * OVERSAMPLE) + 0.5) / OVERSAMPLE


def synthetic_usaf(rng: np.random.Generator, angle_degrees: float = 0.0) -> tuple[np.ndarray, dict]:
    """Bar elements of groups 4-8 rotated by one shared angle, plus flat black and white.

    Each element sits at a random sub-pixel position. Its rect is the integer bounding box
    of the rotated element, as a hand-drawn or registered template would give.
    """
    angle = np.radians(angle_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    tiles = []
    for group in GROUPS:
        for element in range(1, 7):
            bar = PIXELS_PER_MM / (2 * line_pairs_per_mm(group, element))
            size = int(np.ceil(5 * bar * (cos + abs(sin)))) + 2 * MARGIN_PX
            grid = subsample_grid(size)
            for orientation in ("X", "Y"):
                cx, cy = size / 2 + rng.uniform(-0.5, 0.5, size=2)
                dx, dy = grid[None, :] - cx, grid[:, None] - cy
                u, v = dx * cos + dy * sin, -dx * sin + dy * cos
                across, along = (u, v) if orientation == "X" else (v, u)
                across = across + 2.5 * bar
                dark = (
                    (across >= 0)
                    & (across < 5 * bar)
                    & (np.abs(along) < 2.5 * bar)
                    & (np.floor(across / bar) % 2 == 0)
                ).astype(np.float64)
                half = 2.5 * bar * (cos + abs(sin))
                rect = (
                    int(np.floor(cx - half)),
                    int(np.floor(cy - half)),
                    int(np.ceil(cx + half)),
                    int(np.ceil(cy + half)),
                )
                tiles.append(
                    (WHITE - (WHITE - BLACK) * render(dark), group, element, orientation, rect)
                )

    width, gap = 1600, 4
    image = np.full((1200, width), WHITE)
    template: dict = {"source_image": {"width": width, "height": 1200}, "bar_rois": []}
    x = y = row_height = 0
    for tile, group, element, orientation, (rx0, ry0, rx1, ry1) in tiles:
        if x + tile.shape[1] > width:
            x, y, row_height = 0, y + row_height + gap, 0
        image[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        template["bar_rois"].append(
            {
                "group": group,
                "element": element,
                "orientation": orientation,
                "rect": {"x0": x + rx0, "y0": y + ry0, "x1": x + rx1, "y1": y + ry1},
            }
        )
        x += tile.shape[1] + gap
        row_height = max(row_height, tile.shape[0])

    y += row_height + gap
    grid = subsample_grid(240)
    square = ((grid >= 20) & (grid < 220)).astype(np.float64)
    image[y : y + 240, :240] = WHITE - (WHITE - BLACK) * render(np.outer(square, square))
    template["normalization_rois"] = {
        "black": {"x0": 50, "y0": y + 50, "x1": 190, "y1": y + 190},
        "white": {"x0": 400, "y0": y + 50, "x1": 540, "y1": y + 190},
    }
    return image, template


class SyntheticUSAFTests(unittest.TestCase):
    """End-to-end MTF on a rendered target whose true MTF is known exactly."""

    angle_degrees = 0.0

    @classmethod
    def setUpClass(cls) -> None:
        image, template = synthetic_usaf(np.random.default_rng(0), cls.angle_degrees)
        cls.report = calculate_mtf_report(image, template)
        cls.points = [
            (
                int(harmonic),
                fitted.roi_profile.frequency_lp_per_mm * harmonic,
                float(np.hypot(sine, cosine) * np.pi * harmonic / 2),
            )
            for fitted in cls.report.fitted_profiles
            for harmonic, sine, cosine in zip(
                fitted.fit.harmonics,
                fitted.fit.sine_coefficients,
                fitted.fit.cosine_coefficients,
                strict=True,
            )
        ]

    def errors(self, *, harmonic: bool) -> np.ndarray:
        return np.array(
            [
                mtf - true_mtf(f, self.angle_degrees)
                for h, f, mtf in self.points
                if (h > 1) == harmonic and true_mtf(f, self.angle_degrees) > 0.05
            ]
        )

    def test_estimates_rotation(self) -> None:
        self.assertAlmostEqual(self.report.rotation_degrees, self.angle_degrees, delta=0.1)

    def test_calibrates_pixel_scale(self) -> None:
        self.assertAlmostEqual(self.report.pixels_per_mm / PIXELS_PER_MM, 1.0, delta=0.002)

    def test_fundamentals_match_true_mtf(self) -> None:
        errors = self.errors(harmonic=False)
        self.assertEqual(len(errors), 60)
        self.assertLess(np.abs(errors).max(), 0.015)

    def test_harmonics_match_true_mtf(self) -> None:
        errors = self.errors(harmonic=True)
        self.assertGreater(len(errors), 30)
        self.assertLess(np.abs(errors).max(), 0.01)


class SyntheticUSAFRotatedOneDegreeTests(SyntheticUSAFTests):
    angle_degrees = 1.0


class SyntheticUSAFRotatedThreeDegreesTests(SyntheticUSAFTests):
    angle_degrees = -3.0


if __name__ == "__main__":
    unittest.main()
