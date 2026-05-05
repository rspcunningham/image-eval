from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

from image_eval.dqe_plot import save_dqe_curve_plot
from image_eval.mtf_results import MTFResult
from image_eval.nps_results import NPSResult


DQE_CSV_COLUMNS = ["LP per MM", "average MTF", "average NPS", "DQE"]


@dataclass(frozen=True)
class DQEResult:
    frequency_lp_per_mm: float
    average_mtf: float
    average_nps: float
    dqe: float


class DQEReport(NamedTuple):
    results: list[DQEResult]


class DQEReportPaths(NamedTuple):
    output_dir: Path
    csv_path: Path
    plot_path: Path


def calculate_dqe_report(
    mtf_results: Sequence[MTFResult],
    nps_results: Sequence[NPSResult],
) -> DQEReport:
    return DQEReport(results=calculate_dqe_results(mtf_results, nps_results))


def calculate_dqe_results(
    mtf_results: Sequence[MTFResult],
    nps_results: Sequence[NPSResult],
) -> list[DQEResult]:
    nps_frequencies, nps_values = _positive_nps_series(nps_results)
    if len(nps_frequencies) < 2:
        return []

    results: list[DQEResult] = []
    for mtf_result in sorted(mtf_results, key=lambda result: result.frequency_lp_per_mm):
        frequency = mtf_result.frequency_lp_per_mm
        if frequency < nps_frequencies[0] or frequency > nps_frequencies[-1]:
            continue

        average_nps = float(np.interp(frequency, nps_frequencies, nps_values))
        if average_nps <= 0 or not np.isfinite(average_nps):
            continue

        average_mtf = mtf_result.average_mtf
        if not np.isfinite(average_mtf):
            continue

        results.append(
            DQEResult(
                frequency_lp_per_mm=frequency,
                average_mtf=average_mtf,
                average_nps=average_nps,
                dqe=float(average_mtf * average_mtf / average_nps),
            )
        )
    return results


def save_dqe_report(report: DQEReport, output_dir: Path) -> DQEReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dqe.csv"
    plot_path = output_dir / "dqe.png"

    save_dqe_results_csv(report.results, csv_path)
    save_dqe_curve_plot(report.results, plot_path)

    return DQEReportPaths(
        output_dir=output_dir,
        csv_path=csv_path,
        plot_path=plot_path,
    )


def save_dqe_results_csv(results: Sequence[DQEResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DQE_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "LP per MM": _format_float(result.frequency_lp_per_mm),
                "average MTF": _format_float(result.average_mtf),
                "average NPS": _format_float(result.average_nps),
                "DQE": _format_float(result.dqe),
            })


def _positive_nps_series(nps_results: Sequence[NPSResult]) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted(
        (result.frequency, result.average_nps)
        for result in nps_results
        if result.average_nps is not None
        and np.isfinite(result.frequency)
        and np.isfinite(result.average_nps)
        and result.average_nps > 0
    )
    return (
        np.array([frequency for frequency, _ in pairs], dtype=np.float64),
        np.array([average_nps for _, average_nps in pairs], dtype=np.float64),
    )


def _format_float(value: float) -> str:
    return f"{value:.12g}"
