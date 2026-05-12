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
from image_eval.square_wave_fit import (
    FittedBarROIProfile,
    SquareWaveFit,
    fit_bar_roi_profiles,
    fit_square_wave_profile,
)
from image_eval.template_io import base_image_path, load_2d_npy, load_template


MTF_CSV_COLUMNS = ["cycles/mm", "orientation", "mtf"]


@dataclass(frozen=True)
class MTFResult:
    cycles_per_mm: float
    orientation: str
    mtf: float


class MTFReport(NamedTuple):
    results: list[MTFResult]
    fitted_profiles: list[FittedBarROIProfile]


class MTFReportPaths(NamedTuple):
    output_dir: Path
    csv_path: Path


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
    return [
        result
        for fitted_profile in fitted_profiles
        for result in fitted_profile_mtf_results(fitted_profile)
    ]


def line_profile_mtf_points(
    profile: np.ndarray,
    fundamental_cycles_per_mm: float,
) -> dict[float, float]:
    return fitted_square_wave_mtf_points(
        fit_square_wave_profile(profile),
        fundamental_cycles_per_mm,
    )


def fitted_profile_mtf_results(fitted_profile: FittedBarROIProfile) -> list[MTFResult]:
    roi = fitted_profile.roi_profile
    return [
        MTFResult(
            cycles_per_mm=cycles_per_mm,
            orientation=roi.orientation,
            mtf=mtf,
        )
        for cycles_per_mm, mtf in fitted_square_wave_mtf_points(
            fitted_profile.fit,
            roi.frequency_lp_per_mm,
        ).items()
    ]


def fitted_square_wave_mtf_points(
    fit: SquareWaveFit,
    fundamental_cycles_per_mm: float,
) -> dict[float, float]:
    return {
        float(fundamental_cycles_per_mm * harmonic): float(
            np.hypot(sine, cosine) * math.pi * harmonic / 2.0
        )
        for harmonic, sine, cosine in zip(
            fit.harmonics,
            fit.sine_coefficients,
            fit.cosine_coefficients,
            strict=True,
        )
    }


def roi_pixels_per_mm(fitted_profile: FittedBarROIProfile) -> float:
    roi = fitted_profile.roi_profile
    return float(roi.frequency_lp_per_mm * len(roi.profile) / fitted_profile.fit.cycles)


def average_pixels_per_mm_from_fits(fitted_profiles: Sequence[FittedBarROIProfile]) -> float:
    values = [roi_pixels_per_mm(fitted_profile) for fitted_profile in fitted_profiles]
    if not values:
        raise ValueError("cannot calculate pixels per millimetre without fitted ROI profiles")
    return float(np.mean(values))


def save_mtf_report(report: MTFReport, output_dir: Path) -> MTFReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mtf.csv"

    save_mtf_results_csv(report.results, csv_path)

    return MTFReportPaths(
        output_dir=output_dir,
        csv_path=csv_path,
    )


def save_mtf_results_csv(results: Sequence[MTFResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MTF_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "cycles/mm": _format_float(result.cycles_per_mm),
                "orientation": result.orientation,
                "mtf": _format_float(result.mtf),
            })


def _format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
