from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

if TYPE_CHECKING:
    from image_eval.dqe_results import DQEResult


def save_dqe_curve_plot(results: Sequence[DQEResult], output_path: Path) -> None:
    figure = plot_dqe_curve(results)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_dqe_curve(results: Sequence[DQEResult]) -> Figure:
    sorted_results = sorted(results, key=lambda result: result.frequency_lp_per_mm)
    frequencies = np.array(
        [result.frequency_lp_per_mm for result in sorted_results],
        dtype=np.float64,
    )
    dqe = np.array([result.dqe for result in sorted_results], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    if results:
        axis.plot(frequencies, dqe, marker="o", linewidth=2.0, label="DQE")
        axis.legend(loc="best")
    axis.set_xlabel("Spatial frequency (lp/mm)")
    axis.set_ylabel("DQE")
    axis.set_title("Detective Quantum Efficiency")
    axis.grid(True, color="#dddddd", linewidth=0.7)
    figure.tight_layout()
    return figure
