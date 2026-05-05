from __future__ import annotations


def line_pairs_per_mm(group: int, element: int) -> float:
    """Return the USAF 1951 spatial frequency for a group/element pair."""
    if not 1 <= element <= 6:
        raise ValueError(f"USAF 1951 element must be between 1 and 6, got {element}")
    return 2 ** (group + (element - 1) / 6)
