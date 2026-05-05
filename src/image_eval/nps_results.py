from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence, cast

import numpy as np

from image_eval.mtf_profiles import normalize_image_intensity
from image_eval.nps_plot import save_nps_curve_plot, save_nps_spectrum_plot
from image_eval.roi import Rect, as_rect, finite_crop_image
from image_eval.template_io import base_image_path, load_2d_npy, load_template


NPS_ROI_NAMES = ("black", "white")
NPS_MAX_RADIAL_FREQUENCY_CYCLES_PER_PIXEL = 0.5
NPS_MAX_RADIAL_BINS = 128


@dataclass(frozen=True)
class SpatialFrequencyCalibration:
    unit: str
    cycles_per_pixel_multiplier: float

    @property
    def csv_column(self) -> str:
        return f"frequency {self.unit}"

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
    plot_path: Path
    spectrum_dir: Path
    spectrum_paths: list[Path]


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
    normalized = normalize_image_intensity(image, template.get("normalization_rois"))
    roi_crops = _normalization_roi_crops(normalized.image, template.get("normalization_rois"))
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
    spectra_by_name = {spectrum.roi_name: spectrum for spectrum in spectra}
    if not spectra:
        raise ValueError("cannot calculate NPS results without spectra")

    reference_frequencies = spectra[0].radial_frequencies
    results: list[NPSResult] = []
    for index, frequency in enumerate(reference_frequencies):
        results.append(
            NPSResult(
                frequency=float(frequency),
                black_nps=_spectrum_value_or_none(spectra_by_name.get("black"), index),
                white_nps=_spectrum_value_or_none(spectra_by_name.get("white"), index),
            )
        )
    return results


def save_nps_report(report: NPSReport, output_dir: Path) -> NPSReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nps.csv"
    plot_path = output_dir / "nps.png"
    spectrum_dir = output_dir / "nps_spectra"
    spectrum_dir.mkdir(parents=True, exist_ok=True)

    save_nps_results_csv(
        report.results,
        csv_path,
        frequency_calibration=report.frequency_calibration,
    )
    save_nps_curve_plot(
        report.results,
        plot_path,
        frequency_unit=report.frequency_calibration.unit,
    )

    spectrum_paths = []
    for spectrum in report.spectra:
        path = spectrum_dir / f"{spectrum.roi_name}_2d.png"
        save_nps_spectrum_plot(
            spectrum,
            path,
            frequency_unit=report.frequency_calibration.unit,
        )
        spectrum_paths.append(path)

    return NPSReportPaths(
        output_dir=output_dir,
        csv_path=csv_path,
        plot_path=plot_path,
        spectrum_dir=spectrum_dir,
        spectrum_paths=spectrum_paths,
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
            })


def nps_csv_columns(frequency_calibration: SpatialFrequencyCalibration) -> list[str]:
    return [frequency_calibration.csv_column, "black NPS", "white NPS"]


def _normalization_roi_crops(
    normalized_image: np.ndarray,
    normalization_rois: Any,
) -> list[tuple[str, Rect, np.ndarray]]:
    if not isinstance(normalization_rois, dict):
        raise ValueError("template does not contain a normalization_rois object")
    normalization_rois = cast(dict[str, Any], normalization_rois)

    roi_crops: list[tuple[str, Rect, np.ndarray]] = []
    for roi_name in NPS_ROI_NAMES:
        label = f"normalization_rois.{roi_name}"
        rect = as_rect(normalization_rois.get(roi_name), label)
        crop = finite_crop_image(normalized_image, rect, label)
        roi_crops.append((roi_name, rect, crop))
    return roi_crops


def _radial_bin_edges_cycles_per_pixel(crops: Sequence[np.ndarray]) -> np.ndarray:
    if not crops:
        raise ValueError("cannot calculate NPS without normalization ROI crops")

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


def _spectrum_value_or_none(spectrum: NPSSpectrum | None, index: int) -> float | None:
    if spectrum is None or index >= len(spectrum.radial_nps):
        return None
    value = float(spectrum.radial_nps[index])
    if not np.isfinite(value):
        return None
    return value


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return _format_float(value)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
