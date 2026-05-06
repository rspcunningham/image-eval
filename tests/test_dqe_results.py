from __future__ import annotations

import unittest

from image_eval.dqe_results import (
    calculate_dqe_results,
)
from image_eval.mtf_results import MTFResult
from image_eval.nps_results import NPSResult


class DQEResultsTests(unittest.TestCase):
    def test_calculates_dqe_at_mtf_frequencies_with_interpolated_nps(self) -> None:
        mtf_results = [
            MTFResult(frequency_lp_per_mm=1.0, x_mtf=0.5, y_mtf=0.7, average_mtf=0.6),
            MTFResult(frequency_lp_per_mm=2.0, x_mtf=0.2, y_mtf=0.4, average_mtf=0.3),
            MTFResult(frequency_lp_per_mm=4.0, x_mtf=0.1, y_mtf=0.2, average_mtf=0.15),
        ]
        nps_results = [
            NPSResult(frequency=0.0, black_nps=1.0, white_nps=1.0, average_nps=1.0),
            NPSResult(frequency=2.0, black_nps=3.0, white_nps=3.0, average_nps=3.0),
            NPSResult(frequency=3.0, black_nps=5.0, white_nps=5.0, average_nps=5.0),
        ]

        results = calculate_dqe_results(mtf_results, nps_results)

        self.assertEqual([result.frequency_lp_per_mm for result in results], [1.0, 2.0])
        self.assertAlmostEqual(results[0].average_nps, 2.0)
        self.assertAlmostEqual(results[0].dqe, 0.6 * 0.6 / 2.0)
        self.assertAlmostEqual(results[1].dqe, 0.3 * 0.3 / 3.0)

    def test_ignores_non_positive_nps_values(self) -> None:
        mtf_results = [
            MTFResult(frequency_lp_per_mm=1.0, x_mtf=0.5, y_mtf=0.5, average_mtf=0.5),
        ]
        nps_results = [
            NPSResult(frequency=0.0, black_nps=0.0, white_nps=0.0, average_nps=0.0),
            NPSResult(frequency=2.0, black_nps=None, white_nps=None, average_nps=None),
        ]

        self.assertEqual(calculate_dqe_results(mtf_results, nps_results), [])


if __name__ == "__main__":
    unittest.main()
