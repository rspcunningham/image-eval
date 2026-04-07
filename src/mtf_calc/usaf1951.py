from __future__ import annotations


def _validate_group(group: int) -> None:
    if group < -2 or group > 7:
        raise ValueError("USAF 1951 group must be between -2 and 7.")


def _validate_element(element: int) -> None:
    if element < 1 or element > 6:
        raise ValueError("USAF 1951 element must be between 1 and 6.")


def element_frequency_lp_per_mm(group: int, element: int) -> float:
    _validate_group(group)
    _validate_element(element)
    return 2 ** (group + (element - 1) / 6)


def element_line_width_um(group: int, element: int) -> float:
    frequency = element_frequency_lp_per_mm(group, element)
    return 1000.0 / (2.0 * frequency)


def roi_slot_metadata(group: int, element: int) -> dict[str, float | int]:
    return {
        "element": element,
        "spatialFrequencyLpPerMm": element_frequency_lp_per_mm(group, element),
        "lineWidthUm": element_line_width_um(group, element),
    }
