from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from image_eval.mtf_results import MTFResult
from image_eval.nps_results import NPSResult


@dataclass(frozen=True)
class DQEResult:
    frequency_lp_per_mm: float
    average_mtf: float
    average_nps: float
    dqe: float


class DQEReport(NamedTuple):
    results: list[DQEResult]


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
