from __future__ import annotations

import unittest

from image_eval.usaf_1951 import line_pairs_per_mm


class LinePairsPerMMTests(unittest.TestCase):
    def test_calculates_usaf_1951_frequency(self) -> None:
        self.assertEqual(line_pairs_per_mm(0, 1), 1)
        self.assertEqual(line_pairs_per_mm(1, 1), 2)
        self.assertEqual(line_pairs_per_mm(-1, 1), 0.5)
        self.assertEqual(line_pairs_per_mm(4, 1), 16)
        self.assertAlmostEqual(line_pairs_per_mm(0, 6), 2 ** (5 / 6))

    def test_rejects_invalid_elements(self) -> None:
        with self.assertRaises(ValueError):
            line_pairs_per_mm(0, 0)
        with self.assertRaises(ValueError):
            line_pairs_per_mm(0, 7)


if __name__ == "__main__":
    unittest.main()
