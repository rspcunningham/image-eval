from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from image_eval.mtf_profiles import NormalizedImage, normalize_image_intensity
from image_eval.roi import Rect, finite_crop_image
from image_eval.template_io import base_image_path, load_2d_npy, load_template


NPS_MAX_RADIAL_FREQUENCY_CYCLES_PER_PIXEL = 0.5
NPS_MAX_RADIAL_BINS = 128


@dataclass(frozen=True)
class SpatialFrequencyCalibration:
    unit: str
    cycles_per_pixel_multiplier: float

    @property
    def csv_column(self) -> str:
        if self.unit == "lp/mm":
            return "LP per MM"
        return f"frequency {self.unit}"

    def convert(self, frequency_cycles_per_pixel: np.ndarray) -> np.ndarray:
        return frequency_cycles_per_pixel * self.cycles_per_pixel_multiplier


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


class NPSReportPaths(NamedTuple):
    output_dir: Path
    csv_path: Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate black and white ROI NPS from a .npy image and visible ROI template."
    )
    parser.add_argument("template_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        template = load_template(args.template_json)
        image = load_2d_npy(base_image_path(args.template_json, template))
        report = calculate_nps_report(image, template)
        save_nps_report(report, args.output_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"calculate-nps: error: {error}", file=sys.stderr)
        return 1

    return 0


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
    height, width = crop.shape

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


def save_nps_report(report: NPSReport, output_dir: Path) -> NPSReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nps.csv"

    save_nps_results_csv(
        report.results,
        csv_path,
        frequency_calibration=report.frequency_calibration,
    )

    return NPSReportPaths(
        output_dir=output_dir,
        csv_path=csv_path,
    )


def save_nps_results_csv(
    results: Sequence[NPSResult],
    output_path: Path,
    *,
    frequency_calibration: SpatialFrequencyCalibration = CYCLES_PER_PIXEL_FREQUENCY,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = nps_csv_columns(frequency_calibration)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({
                columns[0]: _format_float(result.frequency),
                "black NPS": _format_optional_float(result.black_nps),
                "white NPS": _format_optional_float(result.white_nps),
                "average NPS": _format_optional_float(result.average_nps),
            })


def nps_csv_columns(frequency_calibration: SpatialFrequencyCalibration) -> list[str]:
    return [frequency_calibration.csv_column, "black NPS", "white NPS", "average NPS"]


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


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return _format_float(value)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
