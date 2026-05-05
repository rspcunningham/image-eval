from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from image_eval.dqe_plot import plot_dqe_curve, save_dqe_curve_plot
from image_eval.dqe_results import DQEResult


class DQEPlotTests(unittest.TestCase):
    def test_saves_dqe_curve_plot_as_png(self) -> None:
        results = [
            DQEResult(frequency_lp_per_mm=1.0, average_mtf=0.5, average_nps=0.25, dqe=1.0),
            DQEResult(frequency_lp_per_mm=2.0, average_mtf=0.4, average_nps=0.2, dqe=0.8),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "dqe.png"

            save_dqe_curve_plot(results, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_dqe_curve_plot_allows_empty_results(self) -> None:
        figure = plot_dqe_curve([])
        try:
            self.assertEqual(figure.axes[0].get_ylabel(), "DQE")
        finally:
            figure.clear()


if __name__ == "__main__":
    unittest.main()
