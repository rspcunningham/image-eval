from __future__ import annotations

import unittest
from types import SimpleNamespace

from image_eval.registration_candidates import _candidate_from_matches


class RegistrationCandidateTests(unittest.TestCase):
    def test_rejects_subject_to_base_scales_above_one(self) -> None:
        subject_points = [
            (10.0, 10.0),
            (30.0, 10.0),
            (50.0, 10.0),
            (70.0, 10.0),
            (10.0, 30.0),
            (30.0, 30.0),
            (50.0, 30.0),
            (70.0, 30.0),
            (10.0, 50.0),
            (30.0, 50.0),
        ]
        base_points = [(1.2 * x + 20.0, 1.2 * y + 20.0) for x, y in subject_points]
        matches = [
            SimpleNamespace(queryIdx=index, trainIdx=index)
            for index in range(len(subject_points))
        ]

        candidate = _candidate_from_matches(
            (200, 200),
            (100, 100),
            [SimpleNamespace(pt=point) for point in subject_points],
            [SimpleNamespace(pt=point) for point in base_points],
            matches,
            source="test",
        )

        self.assertIsNone(candidate)


if __name__ == "__main__":
    unittest.main()
