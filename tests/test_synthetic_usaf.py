from __future__ import annotations

import unittest

import numpy as np

from image_eval.mtf_results import calculate_mtf_report
from image_eval.usaf_1951 import line_pairs_per_mm

PIXELS_PER_MM = 1092.0
SIGMA_PX = 0.8  # Gaussian PSF
OVERSAMPLE = 16
BLACK, WHITE = 0.05, 0.95
MARGIN_PX = 12
GROUPS = range(4, 9)


def true_mtf(cycles_per_mm: float) -> float:
    """Gaussian PSF times the pixel's own box aperture."""
    f = cycles_per_mm / PIXELS_PER_MM
    return float(np.exp(-2 * np.pi**2 * SIGMA_PX**2 * f**2) * np.sinc(f))


def blurred_pixels(intervals: list[tuple[float, float]], size_px: int) -> np.ndarray:
    """1-D indicator of the intervals (pixel units), Gaussian-blurred, then pixel-integrated."""
    x = (np.arange(size_px * OVERSAMPLE) + 0.5) / OVERSAMPLE
    signal = np.zeros_like(x)
    for start, stop in intervals:
        signal[(x >= start) & (x < stop)] = 1.0
    radius = int(np.ceil(4 * SIGMA_PX * OVERSAMPLE))
    taps = np.arange(-radius, radius + 1) / OVERSAMPLE
    kernel = np.exp(-0.5 * (taps / SIGMA_PX) ** 2)
    signal = np.convolve(signal, kernel / kernel.sum(), mode="same")
    return signal.reshape(size_px, OVERSAMPLE).mean(axis=1)


def synthetic_usaf(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Bar elements of groups 4-8 with random sub-pixel phase, plus flat black and white.

    Each element is a product of two 1-D masks and the Gaussian PSF and pixel aperture
    are separable, so every tile is exact from two 1-D profiles.
    """
    tiles = []
    for group in GROUPS:
        for element in range(1, 7):
            bar = PIXELS_PER_MM / (2 * line_pairs_per_mm(group, element))
            size = int(np.ceil(5 * bar)) + 2 * MARGIN_PX
            for orientation in ("X", "Y"):
                start_along, start_across = MARGIN_PX + rng.uniform(0, 1, size=2)
                along = blurred_pixels(
                    [
                        (start_along + 2 * k * bar, start_along + (2 * k + 1) * bar)
                        for k in range(3)
                    ],
                    size,
                )
                across = blurred_pixels([(start_across, start_across + 5 * bar)], size)
                dark = np.outer(across, along) if orientation == "X" else np.outer(along, across)
                x0, y0 = (
                    (start_along, start_across)
                    if orientation == "X"
                    else (start_across, start_along)
                )
                rect = (int(x0), int(y0), int(np.ceil(x0 + 5 * bar)), int(np.ceil(y0 + 5 * bar)))
                tiles.append((WHITE - (WHITE - BLACK) * dark, group, element, orientation, rect))

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
    square = blurred_pixels([(20.0, 220.0)], 240)
    image[y : y + 240, :240] = WHITE - (WHITE - BLACK) * np.outer(square, square)
    template["normalization_rois"] = {
        "black": {"x0": 50, "y0": y + 50, "x1": 190, "y1": y + 190},
        "white": {"x0": 400, "y0": y + 50, "x1": 540, "y1": y + 190},
    }
    return image, template


class SyntheticUSAFTests(unittest.TestCase):
    """End-to-end MTF on a rendered target whose true MTF is known exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        image, template = synthetic_usaf(np.random.default_rng(0))
        cls.report = calculate_mtf_report(image, template)
        cls.points = [
            (
                fitted.roi_profile.group,
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

    def errors(self, *, harmonic: bool, groups: range = GROUPS) -> np.ndarray:
        return np.array(
            [
                mtf - true_mtf(f)
                for group, h, f, mtf in self.points
                if (h > 1) == harmonic and group in groups and true_mtf(f) > 0.05
            ]
        )

    def test_calibrates_pixel_scale(self) -> None:
        self.assertAlmostEqual(self.report.pixels_per_mm / PIXELS_PER_MM, 1.0, delta=0.002)

    def test_harmonics_match_true_mtf(self) -> None:
        errors = self.errors(harmonic=True)
        self.assertGreater(len(errors), 30)
        self.assertLess(np.abs(errors).max(), 0.03)

    def test_large_roi_fundamentals_match_true_mtf(self) -> None:
        errors = self.errors(harmonic=False, groups=range(4, 6))
        self.assertLess(np.abs(errors).max(), 0.03)

    def test_small_roi_fundamentals_read_low(self) -> None:
        # Known bias: a 7-22 sample profile spends up to a pixel at each end on white
        # margin, pulling the fundamental low. Tighten this once the fit models it.
        errors = self.errors(harmonic=False, groups=range(6, 9))
        self.assertLess(errors.mean(), 0.0)
        self.assertLess(np.abs(errors).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
