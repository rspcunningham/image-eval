from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np

from image_eval.mtf_profiles import NormalizedImage, normalize_image_intensity
from image_eval.roi import Rect, finite_crop_image


NPS_MAX_RADIAL_FREQUENCY_CYCLES_PER_PIXEL = 0.5
NPS_MAX_RADIAL_BINS = 128


@dataclass(frozen=True)
class SpatialFrequencyCalibration:
    unit: str
    cycles_per_pixel_multiplier: float

    def convert(self, frequency_cycles_per_pixel: np.ndarray) -> np.ndarray:
        scale = self.cycles_per_pixel_multiplier
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("cycles_per_pixel_multiplier must be positive and finite")
        return frequency_cycles_per_pixel * scale


CYCLES_PER_PIXEL_FREQUENCY = SpatialFrequencyCalibration(
    unit="cycles per pixel",
    cycles_per_pixel_multiplier=1.0,
)


@dataclass(frozen=True)
class NPSResult:
    frequency: float
    black_nps: float | None
    white_nps: float | None
    average_nps: float | None


@dataclass(frozen=True)
class NPSSpectrum:
    roi_name: str
    rect: Rect
    crop_mean: float
    crop_variance: float
    frequency_x: np.ndarray
    frequency_y: np.ndarray
    power_spectrum: np.ndarray
    radial_frequencies: np.ndarray
    radial_nps: np.ndarray
    radial_sample_counts: np.ndarray


class NPSReport(NamedTuple):
    results: list[NPSResult]
    spectra: list[NPSSpectrum]
    frequency_calibration: SpatialFrequencyCalibration


def calculate_nps_results(
    image: np.ndarray,
    template: dict[str, Any],
    *,
    frequency_calibration: SpatialFrequencyCalibration = CYCLES_PER_PIXEL_FREQUENCY,
) -> list[NPSResult]:
    return calculate_nps_report(
        image,
        template,
        frequency_calibration=frequency_calibration,
    ).results


def calculate_nps_report(
    image: np.ndarray,
    template: dict[str, Any],
    *,
    frequency_calibration: SpatialFrequencyCalibration = CYCLES_PER_PIXEL_FREQUENCY,
) -> NPSReport:
    normalized = normalize_image_intensity(image, template["normalization_rois"])
    roi_crops = _normalization_roi_crops(normalized)
    radial_edges = _radial_bin_edges_cycles_per_pixel([crop for _, _, crop in roi_crops])

    spectra = [
        calculate_nps_spectrum(
            roi_name,
            rect,
            crop,
            radial_bin_edges_cycles_per_pixel=radial_edges,
            frequency_calibration=frequency_calibration,
        )
        for roi_name, rect, crop in roi_crops
    ]
    return NPSReport(
        results=nps_results_from_spectra(spectra),
        spectra=spectra,
        frequency_calibration=frequency_calibration,
    )


def calculate_nps_spectrum(
    roi_name: str,
    rect: Rect,
    crop: np.ndarray,
    *,
    radial_bin_edges_cycles_per_pixel: np.ndarray,
    frequency_calibration: SpatialFrequencyCalibration = CYCLES_PER_PIXEL_FREQUENCY,
) -> NPSSpectrum:
    crop = np.asarray(crop, dtype=np.float64)
    if crop.ndim != 2:
        raise ValueError(f"{roi_name} NPS ROI is {crop.ndim}D; expected a 2D crop")
    height, width = crop.shape
    if height < 2 or width < 2:
        raise ValueError(f"{roi_name} NPS ROI must be at least 2x2 pixels")
    if not np.all(np.isfinite(crop)):
        raise ValueError(f"{roi_name} NPS ROI must contain only finite pixels")

    crop_mean = float(np.mean(crop))
    noise = crop - crop_mean
    power_spectrum = np.fft.fftshift(np.abs(np.fft.fft2(noise)) ** 2 / noise.size)

    frequency_x_cycles_per_pixel = np.fft.fftshift(np.fft.fftfreq(width))
    frequency_y_cycles_per_pixel = np.fft.fftshift(np.fft.fftfreq(height))
    radial_nps, radial_sample_counts = _radial_average(
        power_spectrum,
        frequency_x_cycles_per_pixel,
        frequency_y_cycles_per_pixel,
        radial_bin_edges_cycles_per_pixel,
    )
    radial_centers_cycles_per_pixel = _bin_centers(radial_bin_edges_cycles_per_pixel)

    return NPSSpectrum(
        roi_name=roi_name,
        rect=dict(rect),
        crop_mean=crop_mean,
        crop_variance=float(np.mean(noise * noise)),
        frequency_x=frequency_calibration.convert(frequency_x_cycles_per_pixel),
        frequency_y=frequency_calibration.convert(frequency_y_cycles_per_pixel),
        power_spectrum=power_spectrum,
        radial_frequencies=frequency_calibration.convert(radial_centers_cycles_per_pixel),
        radial_nps=radial_nps,
        radial_sample_counts=radial_sample_counts,
    )


def nps_results_from_spectra(spectra: Sequence[NPSSpectrum]) -> list[NPSResult]:
    black_spectrum, white_spectrum = spectra

    results: list[NPSResult] = []
    for index, frequency in enumerate(black_spectrum.radial_frequencies):
        black_nps = _spectrum_value_or_none(black_spectrum, index)
        white_nps = _spectrum_value_or_none(white_spectrum, index)
        results.append(
            NPSResult(
                frequency=float(frequency),
                black_nps=black_nps,
                white_nps=white_nps,
                average_nps=_mean_available(black_nps, white_nps),
            )
        )
    return results


def _normalization_roi_crops(
    normalized: NormalizedImage,
) -> tuple[tuple[str, Rect, np.ndarray], tuple[str, Rect, np.ndarray]]:
    image = normalized.image
    rois = normalized.normalization_rois
    return (
        _normalization_roi_crop(image, rois.black, "black"),
        _normalization_roi_crop(image, rois.white, "white"),
    )


def _normalization_roi_crop(
    image: np.ndarray,
    rect: Rect,
    name: str,
) -> tuple[str, Rect, np.ndarray]:
    return name, rect, finite_crop_image(image, rect, f"normalization_rois.{name}")


def _radial_bin_edges_cycles_per_pixel(crops: Sequence[np.ndarray]) -> np.ndarray:
    min_dimension = min(min(crop.shape) for crop in crops)
    if min_dimension < 2:
        raise ValueError("NPS normalization ROIs must be at least 2x2 pixels")

    bin_count = min(NPS_MAX_RADIAL_BINS, min_dimension // 2)
    return np.linspace(
        0.0,
        NPS_MAX_RADIAL_FREQUENCY_CYCLES_PER_PIXEL,
        bin_count + 1,
        dtype=np.float64,
    )


def _radial_average(
    power_spectrum: np.ndarray,
    frequency_x_cycles_per_pixel: np.ndarray,
    frequency_y_cycles_per_pixel: np.ndarray,
    radial_bin_edges_cycles_per_pixel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_x, frequency_y = np.meshgrid(
        frequency_x_cycles_per_pixel,
        frequency_y_cycles_per_pixel,
    )
    radial_frequency = np.sqrt(frequency_x * frequency_x + frequency_y * frequency_y)

    flattened_power = power_spectrum.ravel()
    flattened_frequency = radial_frequency.ravel()
    bin_indices = np.digitize(flattened_frequency, radial_bin_edges_cycles_per_pixel) - 1
    bin_count = len(radial_bin_edges_cycles_per_pixel) - 1

    radial_nps = np.full(bin_count, np.nan, dtype=np.float64)
    sample_counts = np.zeros(bin_count, dtype=np.int64)
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        sample_counts[bin_index] = int(np.count_nonzero(mask))
        if sample_counts[bin_index] > 0:
            radial_nps[bin_index] = float(np.mean(flattened_power[mask]))

    return radial_nps, sample_counts


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("radial bin edges must be a 1D array with at least two values")
    return (edges[:-1] + edges[1:]) / 2.0


def _spectrum_value_or_none(spectrum: NPSSpectrum, index: int) -> float | None:
    value = float(spectrum.radial_nps[index])
    if not np.isfinite(value):
        return None
    return value


def _mean_available(*values: float | None) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return float(np.mean(available))

