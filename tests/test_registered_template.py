from __future__ import annotations

import unittest

from image_eval.registered_template import project_template_rois


class ProjectTemplateROIsTests(unittest.TestCase):
    def test_identity_transform_preserves_visible_rois(self) -> None:
        template = {
            "normalization_rois": {
                "black": {"x0": 1, "y0": 2, "x1": 5, "y1": 7},
                "white": None,
            },
            "bar_rois": [
                {
                    "group": 4,
                    "element": 1,
                    "orientation": "X",
                    "rect": {"x0": 10, "y0": 20, "x1": 30, "y1": 50},
                }
            ],
        }

        output = project_template_rois(
            template,
            [[1, 0, 0], [0, 1, 0]],
            (100, 200),
        )

        self.assertNotIn("base_image_path", output)
        self.assertNotIn("path", output["source_image"])
        self.assertEqual(output["source_image"]["width"], 200)
        self.assertEqual(output["source_image"]["height"], 100)
        self.assertEqual(
            output["normalization_rois"]["black"],
            {"x0": 1, "y0": 2, "x1": 5, "y1": 7},
        )
        self.assertIsNone(output["normalization_rois"]["white"])
        self.assertEqual(
            output["bar_rois"],
            [
                {
                    "group": 4,
                    "element": 1,
                    "orientation": "X",
                    "rect": {"x0": 10, "y0": 20, "x1": 30, "y1": 50},
                }
            ],
        )

    def test_inverts_subject_to_base_transform_before_projecting(self) -> None:
        template = {
            "normalization_rois": {"black": None, "white": None},
            "bar_rois": [
                {
                    "group": 5,
                    "element": 2,
                    "orientation": "Y",
                    "rect": {"x0": 10, "y0": 20, "x1": 30, "y1": 50},
                }
            ],
        }

        output = project_template_rois(
            template,
            [[2, 0, 10], [0, 2, 20]],
            (100, 100),
        )

        self.assertEqual(
            output["bar_rois"][0]["rect"],
            {"x0": 0, "y0": 0, "x1": 10, "y1": 15},
        )

    def test_omits_partial_and_out_of_frame_bar_rois(self) -> None:
        template = {
            "normalization_rois": {"black": None, "white": None},
            "bar_rois": [
                {
                    "group": 4,
                    "element": 1,
                    "orientation": "X",
                    "rect": {"x0": 10, "y0": 10, "x1": 20, "y1": 20},
                },
                {
                    "group": 4,
                    "element": 2,
                    "orientation": "X",
                    "rect": {"x0": -1, "y0": 10, "x1": 20, "y1": 20},
                },
                {
                    "group": 4,
                    "element": 3,
                    "orientation": "X",
                    "rect": {"x0": 90, "y0": 10, "x1": 101, "y1": 20},
                },
            ],
        }

        output = project_template_rois(
            template,
            [[1, 0, 0], [0, 1, 0]],
            (100, 100),
        )

        self.assertEqual(len(output["bar_rois"]), 1)
        self.assertEqual(output["bar_rois"][0]["element"], 1)

    def test_out_of_frame_normalization_rois_become_null(self) -> None:
        template = {
            "normalization_rois": {
                "black": {"x0": 95, "y0": 95, "x1": 101, "y1": 101},
                "white": {"x0": 1, "y0": 1, "x1": 5, "y1": 5},
            },
            "bar_rois": [],
        }

        output = project_template_rois(
            template,
            [[1, 0, 0], [0, 1, 0]],
            (100, 100),
        )

        self.assertIsNone(output["normalization_rois"]["black"])
        self.assertEqual(
            output["normalization_rois"]["white"],
            {"x0": 1, "y0": 1, "x1": 5, "y1": 5},
        )


if __name__ == "__main__":
    unittest.main()
