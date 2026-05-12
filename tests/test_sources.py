from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.sources import load_image_source, load_template_source


class SourceLoadingTests(unittest.TestCase):
    def test_loads_local_path_and_file_url_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            template_path = root / "template.json"
            np.save(image_path, np.zeros((2, 3), dtype=np.float64))
            template_path.write_text(json.dumps({"source_image": {"width": 3, "height": 2}}))

            self.assertEqual(load_image_source(str(image_path)).shape, (2, 3))
            self.assertEqual(load_image_source(image_path.as_uri()).shape, (2, 3))
            self.assertEqual(
                load_template_source(template_path.as_uri())["source_image"],
                {"width": 3, "height": 2},
            )


if __name__ == "__main__":
    unittest.main()
