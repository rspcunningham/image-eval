import json
from pathlib import Path
from typing import Any, Literal, cast

from image_eval.models import Rect

NormName = Literal["black", "white"]


def new_template(
    *,
    source_path: str | Path,
    width: int,
    height: int,
    groups: list[int],
    elements: list[int],
    orientations: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "coordinate_model": "axis_aligned_image_pixels",
        "source_image": {
            "path": str(source_path),
            "width": int(width),
            "height": int(height),
        },
        "groups": [int(group) for group in groups],
        "elements": [int(element) for element in elements],
        "orientations": [str(orientation) for orientation in orientations],
        "anchor": None,
        "normalization_rois": {
            "black": None,
            "white": None,
        },
        "bar_rois": [],
    }


def load_template(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("Template root must be a JSON object.")
    return cast(dict[str, Any], payload)


def save_template(template: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(template, file, indent=2)
        file.write("\n")
    temporary_path.replace(output_path)


def get_anchor_rect(template: dict[str, Any]) -> Rect | None:
    return _optional_rect(template.get("anchor"))


def set_anchor_rect(template: dict[str, Any], rect: Rect) -> None:
    template["anchor"] = rect.to_json()


def get_norm_rect(template: dict[str, Any], name: NormName) -> Rect | None:
    norm_rois = _as_dict(template.setdefault("normalization_rois", {}))
    return _optional_rect(norm_rois.get(name))


def set_norm_rect(template: dict[str, Any], name: NormName, rect: Rect) -> None:
    norm_rois = _as_dict(template.setdefault("normalization_rois", {}))
    norm_rois[name] = rect.to_json()


def get_bar_rect(
    template: dict[str, Any],
    *,
    group: int,
    element: int,
    orientation: str,
) -> Rect | None:
    entry = _find_bar_entry(
        template,
        group=group,
        element=element,
        orientation=orientation,
    )
    if entry is None:
        return None
    return _optional_rect(entry.get("rect"))


def set_bar_rect(
    template: dict[str, Any],
    *,
    group: int,
    element: int,
    orientation: str,
    rect: Rect,
) -> None:
    entry = _find_bar_entry(
        template,
        group=group,
        element=element,
        orientation=orientation,
    )
    if entry is None:
        entry = cast(
            dict[str, Any],
            {
                "group": int(group),
                "element": int(element),
                "orientation": str(orientation),
                "rect": None,
            },
        )
        _bar_rois(template).append(entry)
    entry["rect"] = rect.to_json()


def iter_template_rects(template: dict[str, Any]) -> list[tuple[str, Rect]]:
    labeled_rects: list[tuple[str, Rect]] = []

    anchor = get_anchor_rect(template)
    if anchor is not None:
        labeled_rects.append(("anchor", anchor))

    for norm_name in ("black", "white"):
        rect = get_norm_rect(template, norm_name)
        if rect is not None:
            labeled_rects.append((f"norm {norm_name}", rect))

    for entry in _bar_rois(template):
        rect = _optional_rect(entry.get("rect"))
        if rect is None:
            continue
        label = f"G{entry.get('group')} E{entry.get('element')} {entry.get('orientation')}"
        labeled_rects.append((label, rect))

    return labeled_rects


def _find_bar_entry(
    template: dict[str, Any],
    *,
    group: int,
    element: int,
    orientation: str,
) -> dict[str, Any] | None:
    for entry in _bar_rois(template):
        if (
            entry.get("group") == group
            and entry.get("element") == element
            and entry.get("orientation") == orientation
        ):
            return entry
    return None


def _bar_rois(template: dict[str, Any]) -> list[dict[str, Any]]:
    bar_rois = template.setdefault("bar_rois", [])
    if not isinstance(bar_rois, list):
        raise TypeError("Template bar_rois must be a list.")
    for entry in bar_rois:
        if not isinstance(entry, dict):
            raise TypeError("Each bar ROI entry must be a JSON object.")
    return cast(list[dict[str, Any]], bar_rois)


def _optional_rect(payload: object) -> Rect | None:
    if payload is None:
        return None
    return Rect.from_json(payload)


def _as_dict(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Expected JSON object.")
    return cast(dict[str, Any], payload)
