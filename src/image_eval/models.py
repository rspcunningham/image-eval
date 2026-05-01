from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def is_valid(self) -> bool:
        return self.width >= 2.0 and self.height >= 2.0

    @classmethod
    def from_bounds(cls, left: float, top: float, right: float, bottom: float) -> "Rect":
        normalized_left = min(left, right)
        normalized_top = min(top, bottom)
        normalized_right = max(left, right)
        normalized_bottom = max(top, bottom)
        return cls(
            left=normalized_left,
            top=normalized_top,
            width=normalized_right - normalized_left,
            height=normalized_bottom - normalized_top,
        )

    @classmethod
    def from_json(cls, payload: object) -> "Rect":
        if not isinstance(payload, dict):
            raise TypeError("Expected rect JSON object.")
        rect_payload = cast(dict[str, object], payload)
        return cls(
            left=_as_float(rect_payload["left"]),
            top=_as_float(rect_payload["top"]),
            width=_as_float(rect_payload["width"]),
            height=_as_float(rect_payload["height"]),
        )

    def to_json(self) -> dict[str, float]:
        return {
            "left": float(self.left),
            "top": float(self.top),
            "width": float(self.width),
            "height": float(self.height),
        }

    def translated(self, dx: float, dy: float) -> "Rect":
        return Rect(
            left=self.left + dx,
            top=self.top + dy,
            width=self.width,
            height=self.height,
        )

    def clamp(self, *, width: int, height: int) -> "Rect":
        rect_width = min(max(self.width, 2.0), float(width))
        rect_height = min(max(self.height, 2.0), float(height))
        left = min(max(self.left, 0.0), max(float(width) - rect_width, 0.0))
        top = min(max(self.top, 0.0), max(float(height) - rect_height, 0.0))
        return Rect(left=left, top=top, width=rect_width, height=rect_height)


def _as_float(value: Any) -> float:
    if not isinstance(value, int | float):
        raise TypeError("Expected number.")
    return float(value)
