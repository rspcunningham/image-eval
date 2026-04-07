import json
import numpy as np
from typing import cast
from numpy.typing import NDArray

from mtf_calc.models import Anchor, BarSection, Dim, NormRegion, Point, Roi, RoiConfig, ScaleGroup

def load_source(path: str) -> NDArray[np.float32]:
    return cast(NDArray[np.float32], np.load(path))


def save_roi_config(config: RoiConfig, path: str) -> None:
    payload: dict[str, object] = {
        "anchor": {
            "roi": _serialize_roi(config.anchor.roi),
        },
        "scale_groups": [int(group) for group in config.scale_groups],
        "bar_rois": [
            {
                "group": section.group,
                "element": section.element,
                "dim": section.dim,
                "roi": _serialize_roi(roi),
            }
            for section, roi in sorted(
                config.bar_rois.items(),
                key=lambda item: (item[0].group, item[0].element, item[0].dim),
            )
        ],
        "norm_rois": {
            region: _serialize_roi(roi)
            for region, roi in config.norm_rois.items()
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        _ = f.write("\n")


def load_roi_config(path: str) -> RoiConfig:
    with open(path, encoding="utf-8") as f:
        payload = _as_object(cast(object, json.load(f)))

    anchor_payload = _as_object(payload["anchor"])
    scale_groups_payload = _as_array(payload["scale_groups"])
    bar_rois_payload = _as_array(payload["bar_rois"])
    norm_rois_payload = _as_object(payload["norm_rois"])

    return RoiConfig(
        anchor=Anchor(roi=_deserialize_roi(_as_object(anchor_payload["roi"]))),
        scale_groups=[
            cast(ScaleGroup, _as_int(group))
            for group in scale_groups_payload
        ],
        bar_rois={
            BarSection(
                group=_as_int(entry["group"]),
                element=_as_int(entry["element"]),
                dim=cast(Dim, _as_str(entry["dim"])),
            ): _deserialize_roi(_as_object(entry["roi"]))
            for entry in (_as_object(item) for item in bar_rois_payload)
        },
        norm_rois={
            cast(NormRegion, region): _deserialize_roi(_as_object(roi_payload))
            for region, roi_payload in norm_rois_payload.items()
        },
    )


def _serialize_roi(roi: Roi) -> dict[str, object]:
    return {
        "top_left": _serialize_point(roi.top_left),
        "top_right": _serialize_point(roi.top_right),
        "bottom_left": _serialize_point(roi.bottom_left),
        "bottom_right": _serialize_point(roi.bottom_right),
    }


def _deserialize_roi(payload: dict[str, object]) -> Roi:
    return Roi(
        top_left=_deserialize_point(_as_object(payload["top_left"])),
        top_right=_deserialize_point(_as_object(payload["top_right"])),
        bottom_left=_deserialize_point(_as_object(payload["bottom_left"])),
        bottom_right=_deserialize_point(_as_object(payload["bottom_right"])),
    )


def _serialize_point(point: Point) -> dict[str, float]:
    return {
        "x": point.x,
        "y": point.y,
    }


def _deserialize_point(payload: dict[str, object]) -> Point:
    return Point(
        x=_as_float(payload["x"]),
        y=_as_float(payload["y"]),
    )


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected object")
    return cast(dict[str, object], value)


def _as_array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("Expected array")
    return cast(list[object], value)


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected string")
    return value


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("Expected integer")
    return value


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected number")
    return float(value)
