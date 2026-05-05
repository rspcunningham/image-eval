from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from image_eval.mtf_profiles import prepare_mtf_profiles
from image_eval.mtf_plot import save_bar_roi_fit_plot, save_mtf_curve_plot
from image_eval.square_wave_fit import FittedBarROIProfile, fit_bar_roi_profiles
from image_eval.template_io import base_image_path, load_2d_npy, load_template


FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION = math.pi / 2.0
MTF_CSV_COLUMNS = ["LP per MM", "XMTF", "YMTF", "average MTF"]


@dataclass(frozen=True)
class MTFResult:
    frequency_lp_per_mm: float
    x_mtf: float | None
    y_mtf: float | None
    average_mtf: float


class MTFReport(NamedTuple):
    results: list[MTFResult]
    fitted_profiles: list[FittedBarROIProfile]


class MTFReportPaths(NamedTuple):
    output_dir: Path
    csv_path: Path
    plot_path: Path
    roi_fit_dir: Path
    roi_fit_paths: list[Path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate USAF 1951 bar-target MTF from a .npy image and visible ROI template."
    )
    parser.add_argument("template_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        template = load_template(args.template_json)
        image = load_2d_npy(base_image_path(args.template_json, template))
        report = calculate_mtf_report(image, template)
        save_mtf_report(report, args.output_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"calculate-mtf: error: {error}", file=sys.stderr)
        return 1

    return 0


def calculate_mtf_results(image: np.ndarray, template: dict[str, Any]) -> list[MTFResult]:
    return calculate_mtf_report(image, template).results


def calculate_mtf_report(image: np.ndarray, template: dict[str, Any]) -> MTFReport:
    prepared = prepare_mtf_profiles(image, template)
    fitted_profiles = fit_bar_roi_profiles(prepared.bar_profiles)
    return MTFReport(
        results=mtf_results_from_fits(fitted_profiles),
        fitted_profiles=fitted_profiles,
    )


def mtf_results_from_fits(fitted_profiles: Sequence[FittedBarROIProfile]) -> list[MTFResult]:
    grouped: dict[float, dict[str, list[float]]] = {}

    for fitted_profile in fitted_profiles:
        roi = fitted_profile.roi_profile
        orientation = roi.orientation
        if orientation not in ("X", "Y"):
            raise ValueError(f"bar ROI orientation must be X or Y, got {orientation}")

        orientation_values = grouped.setdefault(
            roi.frequency_lp_per_mm,
            {"X": [], "Y": []},
        )
        orientation_values[orientation].append(roi_mtf_value(fitted_profile))

    results: list[MTFResult] = []
    for frequency, orientation_values in sorted(grouped.items()):
        x_mtf = _mean_or_none(orientation_values["X"])
        y_mtf = _mean_or_none(orientation_values["Y"])
        average_mtf = _mean_available(x_mtf, y_mtf)
        results.append(
            MTFResult(
                frequency_lp_per_mm=frequency,
                x_mtf=x_mtf,
                y_mtf=y_mtf,
                average_mtf=average_mtf,
            )
        )

    return results


def roi_mtf_value(fitted_profile: FittedBarROIProfile) -> float:
    return fitted_profile.fit.fundamental_amplitude * FUNDAMENTAL_TO_SQUARE_WAVE_MODULATION


def roi_pixels_per_mm(fitted_profile: FittedBarROIProfile) -> float:
    roi = fitted_profile.roi_profile
    profile_pixels = len(roi.profile)
    fitted_cycles = fitted_profile.fit.cycles
    known_lp_per_mm = roi.frequency_lp_per_mm

    if profile_pixels <= 0:
        raise ValueError("ROI profile must contain at least one pixel")
    if not np.isfinite(fitted_cycles) or fitted_cycles <= 0:
        raise ValueError("fitted ROI cycles must be positive and finite")
    if not np.isfinite(known_lp_per_mm) or known_lp_per_mm <= 0:
        raise ValueError("ROI frequency_lp_per_mm must be positive and finite")

    return float(known_lp_per_mm * profile_pixels / fitted_cycles)


def average_pixels_per_mm_from_fits(fitted_profiles: Sequence[FittedBarROIProfile]) -> float:
    values = [roi_pixels_per_mm(fitted_profile) for fitted_profile in fitted_profiles]
    if not values:
        raise ValueError("cannot calculate pixels per millimetre without fitted ROI profiles")
    return float(np.mean(values))


def save_mtf_report(report: MTFReport, output_dir: Path) -> MTFReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mtf.csv"
    plot_path = output_dir / "mtf.png"
    roi_fit_dir = output_dir / "roi_fits"
    roi_fit_dir.mkdir(parents=True, exist_ok=True)

    save_mtf_results_csv(report.results, csv_path)
    save_mtf_curve_plot(report.results, plot_path)

    roi_fit_paths = []
    for index, fitted_profile in enumerate(report.fitted_profiles, start=1):
        roi = fitted_profile.roi_profile
        path = roi_fit_dir / (
            f"{index:03d}_g{roi.group}_e{roi.element}_{roi.orientation.lower()}_fit.png"
        )
        save_bar_roi_fit_plot(
            fitted_profile,
            path,
            mtf_value=roi_mtf_value(fitted_profile),
        )
        roi_fit_paths.append(path)

    return MTFReportPaths(
        output_dir=output_dir,
        csv_path=csv_path,
        plot_path=plot_path,
        roi_fit_dir=roi_fit_dir,
        roi_fit_paths=roi_fit_paths,
    )


def save_mtf_results_csv(results: Sequence[MTFResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MTF_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "LP per MM": _format_float(result.frequency_lp_per_mm),
                "XMTF": _format_optional_float(result.x_mtf),
                "YMTF": _format_optional_float(result.y_mtf),
                "average MTF": _format_float(result.average_mtf),
            })


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _mean_available(*values: float | None) -> float:
    available = [value for value in values if value is not None]
    if not available:
        raise ValueError("cannot calculate average MTF without X or Y values")
    return float(np.mean(available))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return _format_float(value)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
