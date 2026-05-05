from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.template_io import load_2d_npy


class TemplateIOTests(unittest.TestCase):
    def test_load_2d_npy_converts_complex_arrays_to_magnitude_squared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.npy"
            np.save(path, np.array([[3 + 4j, 1 - 2j]], dtype=np.complex64))

            image = load_2d_npy(path)

            self.assertFalse(np.iscomplexobj(image))
            np.testing.assert_allclose(image, np.array([[25, 5]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
